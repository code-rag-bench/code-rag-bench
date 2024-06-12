from datasets import load_dataset
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import voyageai
import numpy as np
import asyncio
import logging
import pathlib, os
import random
import math
import json

import argparse
from utils import BEIR_DATASETS

from tqdm import tqdm

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from transformers import AutoTokenizer


MODEL2MAXLEN = {"voyage-2": 4000,
                "voyage-large-2": 16000,
                "voyage-code-2": 16000,
                "voyage-large-2-instruct": 16000,
                "voyage-law":16000}
LIMIT=120000

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6))
def embed_with_backoff(retriever, **kwargs):
    return retriever.embed(**kwargs)


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6))
async def async_with_backoff(retriever, **kwargs):
    response = [retriever.embed(**kwargs)]
    return await asyncio.gather(*response)


def get_voyage_document_embeddings(corpus, retriever, args):
    documents, doc_ids = [], []
    for doc_id in corpus:
        doc = corpus[doc_id]
        documents.append(doc["text"])
        doc_ids.append(doc_id)

    documents_embeddings = []
    # Generate embeddings in batches
    for i in tqdm(range(0, len(documents), args.batch_size)):
        end = min(len(documents), i + args.batch_size)
        batch = documents[i:end]
        num_tokens = retriever.count_tokens(batch)
        if num_tokens >= LIMIT:
            min_split = math.ceil(num_tokens / LIMIT)
            dynamic_batch_size = args.batch_size // min_split
            logging.info(f"Batch {i} has {num_tokens} tokens, splitting into {min_split} requests")
            for split in range(min_split):
                split_start = split * dynamic_batch_size
                split_end = min(len(batch), (split + 1) * dynamic_batch_size)
                split_batch = batch[split_start:split_end]
                curr_split_tokens = retriever.count_tokens(split_batch)
                if curr_split_tokens >= LIMIT and len(split_batch) >= 1:
                    for doc in split_batch:
                        if retriever.count_tokens([doc]) == 0:
                            documents_embeddings.append([])
                            continue
                        if args.run_async:
                            split_embeddings = asyncio.run(
                                async_with_backoff(retriever=retriever, texts=doc, model=args.model, input_type="document", truncation=True))
                            split_embeddings = split_embeddings[0].embeddings
                        else:
                            split_embeddings = embed_with_backoff(retriever=retriever, texts=doc, model=args.model, input_type="document",
                                                                        truncation=True).embeddings
                        documents_embeddings.extend(split_embeddings)
                elif len(split_batch) == 0:
                    continue
                else:
                    if args.run_async:
                        split_embeddings = asyncio.run(
                            async_with_backoff(retriever=retriever, texts=split_batch, model=args.model, input_type="document", truncation=True))
                        split_embeddings = split_embeddings[0].embeddings
                    else:
                        split_embeddings = embed_with_backoff(retriever=retriever, texts=split_batch, model=args.model, input_type="document",
                                                                    truncation=True).embeddings
                    documents_embeddings.extend(split_embeddings)
        else:
            # Generate embeddings for current batch
            if retriever.count_tokens(batch) == 0:
                documents_embeddings.extend([] * len(batch))
                continue
            if args.run_async:
                batch_embeddings = asyncio.run(async_with_backoff(retriever=retriever, texts=batch, model=args.model, input_type="document", truncation=True))
                batch_embeddings = batch_embeddings[0].embeddings
            else:
                batch_embeddings = embed_with_backoff(retriever=retriever, texts=batch, model=args.model, input_type="document", truncation=True).embeddings
            # Add to the list of embeddings
            documents_embeddings.extend(batch_embeddings)

    return documents_embeddings, doc_ids


def get_voyage_query_embeddings(queries, retriever, args):
    sorted_queries = []
    queryidx2sortedidx = dict()  # bookkeeping orders although python internal hashtable should preserve the order
    sorted_query_idx = 0
    for query_id in queries:
        sorted_queries.append(queries[query_id])
        queryidx2sortedidx[query_id] = sorted_query_idx
        sorted_query_idx += 1
    assert len(sorted_queries) == len(
        queries), f"length of sorted_queries should be equal to length of queries, now is {len(sorted_queries)} and {len(queries)}"

    query_embeddings = []

    # Generate embeddings in batches
    for i in tqdm(range(0, len(queries), args.batch_size)):
        end = min(len(queries), i + args.batch_size)
        batch = sorted_queries[i:end]
        if retriever.count_tokens(batch) == 0:
            query_embeddings.extend([] * len(batch))
            continue
        if args.run_async:
            batch_embeddings = asyncio.run(async_with_backoff(retriever=retriever, texts=batch, model=args.model, input_type="query"))
            assert len(
                batch_embeddings) == 1, f"length of async batch_embeddings should be 1, now is {len(batch_embeddings)}"
            batch_embeddings = batch_embeddings[0].embeddings
        else:
            batch_embeddings = embed_with_backoff(retriever=retriever, texts=batch, model=args.model, input_type="query").embeddings
        query_embeddings.extend(batch_embeddings)

    return query_embeddings, queryidx2sortedidx


def get_top_docs(results, corpus, task_id: str, topk:int=10) -> list[str]:
    if task_id not in results:
        return []
    doc_scores = results[task_id]
    doc_scores_sorted = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    doc_scores_sorted = doc_scores_sorted[:topk]
    doc_code_snippets = [corpus[code_id] for code_id, score in doc_scores_sorted]
    return doc_code_snippets


def main():
#### /print debug information to stdout

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="code_search_net_python",
                        choices=["humaneval", "mbpp", "apps", "code_search_net_python", "ds1000_all_completion","swe-bench-lite", "livecodebench",
                                 "odex_en", "odex_es", "odex_ja", "odex_ru", "docprompting_conala", "repoeval/function",],
                        help="Dataset to use for evaluation")
    parser.add_argument("--model", type=str, default="voyage-code-2", help="Sentence-BERT model to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Sentence-BERT model to use, for ds1000_all_completion, use 16")
    parser.add_argument("--output_file", type=str, required=True, default=None, help="Specify the filepath if you want to save the retrieval (evaluation) results.")
    parser.add_argument("--results_file", type=str, required=True, default=None, help="Specify the filepath if you want to save the retrieval results.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the repoeval dataset.")
    parser.add_argument("--api_key_fp", type=str, default='../api_keys/voyager_ai.txt', help="API key file name. There SHOULD be a newline at the end of the file.")
    parser.add_argument("--run_async", action="store_true", help="Use async calls to the API.")
    args = parser.parse_args()

    assert args.results_file.endswith('.jsonl'), "For HF dataset purposes: please provide a .jsonl file for the results file."
    dataset = args.dataset
    # api_key check
    if args.api_key_fp is not None:
        with open(args.api_key_fp) as f:
            api_key = f.read()[:-1]
    elif 'VOYAGE_API_KEY' not in os.environ:
        raise ValueError("Please set environmental variable VOYAGE_API_KEY to your API key, or pass in --api_key_fp")
    else:
        api_key = os.environ['VOYAGE_API_KEY']

    #### Download nfcorpus.zip dataset and unzip the dataset
    if dataset in BEIR_DATASETS:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
    else:
        corpus, queries, qrels = GenericDataLoader(data_folder=os.path.join("datasets", dataset)).load(split="test")

    if args.run_async:
        retriever = voyageai.AsyncClient(api_key=api_key)
    else:
        retriever = voyageai.Client(api_key=api_key)


    doc_embedding_path = os.path.join("datasets", dataset, "voyage_doc_embeddings.npy")
    doc_ids_path = os.path.join("datasets", dataset, "voyage_doc_ids.json")

    if os.path.exists(doc_embedding_path) and os.path.exists(doc_ids_path):
        documents_embeddings_np = np.load(doc_embedding_path)
        documents_embeddings = documents_embeddings_np.tolist()
        with open(doc_ids_path, "r") as f:
            doc_ids = json.load(f)
    else:
        documents_embeddings, doc_ids = get_voyage_document_embeddings(corpus, retriever, args)
        documents_embeddings_np = np.array(documents_embeddings)
        np.save(doc_embedding_path, documents_embeddings_np)
        with open(doc_ids_path, "w+") as f:
            json.dump(doc_ids, f)

    query_embedding_path = os.path.join("datasets", dataset, "voyage_query_embeddings.npy")
    queryidx2sortedidx_path = os.path.join("datasets", dataset, "voyage_queryidx2truncatedidx.json")

    if os.path.exists(query_embedding_path) and os.path.exists(queryidx2sortedidx_path):
        query_embeddings_np = np.load(query_embedding_path)
        query_embeddings = query_embeddings_np.tolist()
        with open(queryidx2sortedidx_path, "r") as f:
            queryidx2sortedidx = json.load(f)
    else:
        query_embeddings, queryidx2sortedidx = get_voyage_query_embeddings(queries, retriever, args)
        query_embeddings_np = np.array(query_embeddings)
        np.save(query_embedding_path, query_embeddings_np)
        with open(queryidx2sortedidx_path, "w+") as f:
            json.dump(queryidx2sortedidx, f)

    results = {}

    assert len(query_embeddings) == len(queries), f"length of query_embeddings should be {len(queries)}, now is {len(query_embeddings)}"
    for query_id in tqdm(queries):
        query_embedding = query_embeddings[queryidx2sortedidx[query_id]]
        similarities = np.dot(documents_embeddings, query_embedding)
        results[query_id] = {}
        for doc_id, score in zip(doc_ids, similarities):
            results[query_id][doc_id] = score

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    if args.dataset != "livecodebench":
        k_values=[1,3,5,10,100]
        model = DRES(models.SentenceBERT("BAAI/bge-base-en-v1.5"), batch_size=args.batch_size, corpus_chunk_size=512*9999)

        retriever = EvaluateRetrieval(model, score_function="dot")

        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)

        mrr = retriever.evaluate_custom(qrels, results, k_values, metric="mrr")
        recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
        hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

        all_results = {"ndcg": ndcg, "mrr": mrr, "recall": recall, "precision": precision}
        print(all_results)
        with open(args.output_file, "w") as f:
            json.dump(all_results, f)

    if args.dataset in ["humaneval", "mbpp", "apps"]:
        if args.dataset == "humaneval":
            ds = load_dataset("openai_humaneval")
            id_key = "task_id"
        elif args.dataset == "mbpp":
            ds = load_dataset("mbpp")
            id_key = "task_id"
        elif args.dataset == "apps":
            ds = load_dataset("codeparrot/apps")
            id_key = "problem_id"
        all_top_docs = []
        for task_id in ds["test"][id_key]:
            all_top_docs.append(get_top_docs(f"{task_id}_doc"))
        ds["test"] = ds["test"].add_column("docs", all_top_docs)
        ds["test"].to_json(args.results_file)  # this outputs to arrow format and read as .jsonl
    elif "odex" in args.dataset:
        lang = args.dataset.split("_")[-1]
        ds = load_dataset("neulab/odex", lang)
        all_top_docs = []
        for idx, task_id in enumerate(ds["test"]["task_id"]):
            all_top_docs.append(get_top_docs(f"{idx}_{task_id}"))
        ds["test"] = ds["test"].add_column("docs", all_top_docs)
        ds["test"].to_json(args.results_file)  # this outputs to arrow format and read as .jsonl
    elif args.dataset == "docprompting_conala":
        ds = load_dataset("neulab/docprompting-conala")
        all_top_docs = []
        for idx, task_id in enumerate(ds["test"]["question_id"]):
            all_top_docs.append(get_top_docs(task_id))
        ds["test"] = ds["test"].add_column("docs", all_top_docs)
        ds["test"].to_json(args.results_file)  # this outputs to arrow format and read as .jsonl
    elif args.dataset.startswith("ds1000"):
        _, key, mode = args.dataset.split("_")
        key = key.capitalize()
        mode = mode.capitalize()
        from create.ds1000 import get_dataset
        source_dir = pathlib.Path(__file__).parent / "ds"
        data = get_dataset(source_dir, mode=mode, key=key)
        all_docs = []
        example_ids = []
        for item in data:
            example = item.data
            example_id = f"{example['lib']}_{example['perturbation_origin_id']}"
            all_docs.append(get_top_docs(example_id))
            example_ids.append(example_id)
        assert len(all_docs) == len(example_ids), f"length of all_docs should be {len(example_ids)}, now is {len(all_docs)}"
        with open(args.results_file, "w+") as fout:
            for idx, all_doc in enumerate(all_docs):
                fout.write(json.dumps({"example_id": example_id,
                                       "docs": all_doc}) + "\n")
    elif args.dataset.startswith("repoeval"):
        tasks = [json.loads(line.strip()) for line in open(args.dataset_path, 'r')]
        prompts, references, docs = [], [], []
        for task in tasks:
            if task["metadata"]["task_id"] not in queries:
                continue
            prompts.append('\n'.join(task["prompt"].split('\n')[-5:]))
            references.append(task["metadata"]["ground_truth"])
            docs.append(get_top_docs(task["metadata"]["task_id"]))
        assert len(prompts) == len(references) == len(docs)
        dataset = [
            {"prompt": p, "reference": r, "docs": d}
            for p,r,d in zip(prompts, references, docs)
        ]
        with open(args.results_file, "w+") as fout:  # preserve jsonl
            for curr in dataset:
                fout.write(json.dumps(curr) + "\n")
    elif args.dataset.startswith("swe-bench"):
        if args.dataset == "swe-bench-lite":
            ds = load_dataset("princeton-nlp/SWE-bench_Lite_oracle")
        else:
            ds = load_dataset("princeton-nlp/SWE-bench_oracle")
        all_top_docs = []
        for instance_id in ds["test"]["instance_id"]:
            all_top_docs.append(get_top_docs(instance_id))
        ds["test"] = ds["test"].add_column("docs", all_top_docs)
        ds["test"].to_json(args.results_file)  # this outputs to arrow format and read as .jsonl
    else:
        with open(args.results_file, 'w+') as fw:
            for curr in results:
                fw.write(json.dumps({curr: results[curr]}) + "\n")


    #### Print top-k documents retrieved ####
    top_k = 3

    query_id, ranking_scores = random.choice(list(results.items()))
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    logging.info("Query : %s\n" % queries[query_id])

    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        # Format: Rank x: ID [Title] Body
        logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))


if __name__ == "__main__":
    main()
