from datasets import load_dataset
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import numpy as np
import tiktoken

import asyncio

import logging
import pathlib, os
import random
import json

import argparse
from utils import BEIR_DATASETS

from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI, AsyncOpenAI
from modify_corpus_for_bm25 import modify_single_dataset

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
embedding_encoding = "cl100k_base"

# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(client, texts, model="text-embedding-3-small") -> list[float]:
    return [out.embedding for out in client.embeddings.create(input=texts, model=model).data]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def async_get_embedding(client, texts, model="text-embedding-3-small") -> list[float]:
    response = [client.embeddings.create(input=texts, model=model)]
    return await asyncio.gather(*response)


def truncate_text_tokens(text, encoding_name=embedding_encoding, max_tokens=7500):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text, allowed_special={'<|endoftext|>'})[:max_tokens]


def get_document_embeddings(corpus, client, args):
    if args.corpus_path is not None:
        documents, doc_ids = [], []
        for doc_id in corpus:
            doc = corpus[doc_id]
            documents.append(truncate_text_tokens(doc))
            doc_ids.append(doc_id)
    else:
        documents, doc_ids = [], []
        for doc_id in corpus:
            doc = corpus[doc_id]
            documents.append(truncate_text_tokens(doc["text"]))
            doc_ids.append(doc_id)


    documents_embeddings = []
    # Generate embeddings in batches
    for i in tqdm(range(0, len(documents), args.batch_size)):
        end = min(len(documents), i + args.batch_size)
        batch = documents[i:end]
        # Generate embeddings for current batch
        if args.run_async:
            batch_embeddings = asyncio.run(async_get_embedding(client, texts=batch, model=args.model))
            assert len(
                batch_embeddings) == 1, f"length of async batch_embeddings should be 1, now is {len(batch_embeddings)}"
            batch_embeddings = [out.embedding for out in batch_embeddings[0].data]
        else:
            batch_embeddings = get_embedding(client, texts=batch, model=args.model)
        # Add to the list of embeddings
        documents_embeddings.extend(batch_embeddings)
    return documents_embeddings, doc_ids


def get_query_embeddings(queries, client, args):
    truncated_queries = []
    queryidx2truncatedidx = dict()  # bookkeeping orders although python internal hashtable should preserve the order
    truncate_queries_idx = 0
    for query_id in queries:
        query = truncate_text_tokens(queries[query_id])
        truncated_queries.append(query)
        queryidx2truncatedidx[query_id] = truncate_queries_idx
        truncate_queries_idx += 1
    assert len(truncated_queries) == len(
        queries), f"length of truncated_queries should be equal to length of queries, now is {len(truncated_queries)} and {len(queries)}"

    query_embeddings = []  # this follows the order of truncated_queries, make sure that the order is correct
    # Generate embeddings in batches
    for i in tqdm(range(0, len(truncated_queries), args.batch_size)):
        end = min(len(truncated_queries), i + args.batch_size)
        batch = truncated_queries[i:end]
        # Generate embeddings for current batch
        if args.run_async:
            batch_embeddings = asyncio.run(async_get_embedding(client, texts=batch, model=args.model))
            assert len(
                batch_embeddings) == 1, f"length of async batch_embeddings should be 1, now is {len(batch_embeddings)}"
            batch_embeddings = [out.embedding for out in batch_embeddings[0].data]
        else:
            batch_embeddings = get_embedding(client, texts=batch, model=args.model)

        # Add to the list of embeddings
        query_embeddings.extend(batch_embeddings)
    return query_embeddings, queryidx2truncatedidx


def main():
#### /print debug information to stdout

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="scifact",
                        choices=["humaneval", "mbpp", "apps", "code_search_net", "swe-bench-lite", "livecodebench", "code_search_net_python",
                                 "odex_en", "odex_es", "odex_ja", "odex_ru", "docprompting_conala", "ds1000_all_completion", "ds1000_all_insertion", "repoeval/function",],
                        help="Dataset to use for evaluation")
    parser.add_argument("--model", type=str, default="text-embedding-3-small", help="Sentence-BERT model to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Sentence-BERT model to use")
    parser.add_argument("--corpus_path", type=str, default=None, help="if specified, meaning that we don't use existing corpus in the dataset, and specified a processed corpus.")
    parser.add_argument("--output_file", type=str, default=None, required=True, help="Specify the filepath if you want to save the retrieval (evaluation) results.")
    parser.add_argument("--results_file", type=str, default=None, required=True, help="Specify the filepath if you want to save the retrieval results.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the repoeval dataset.")
    parser.add_argument("--api_key_fp", type=str, default=None, help="API key file name. There SHOULD be a newline at the end of the file.")
    parser.add_argument("--run_async", action="store_true", help="Use async calls to the API.")
    args = parser.parse_args()

    # assert args.results_file.endswith('.json'), "For HF dataset purposes: please provide a .json file for the results file."


    if args.dataset.startswith("repoeval"):
        assert args.dataset_path is not None, "Please provide the path to the repoeval dataset."
        assert args.dataset_path.endswith(".jsonl"), "The path to the repoeval dataset should end in .jsonl"

    dataset = args.dataset

    #### Download nfcorpus.zip dataset and unzip the dataset
    if dataset in BEIR_DATASETS:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
    else:
        corpus, queries, qrels = GenericDataLoader(data_folder=os.path.join("datasets", dataset)).load(split="test")


    if args.api_key_fp is not None:
        with open(args.api_key_fp) as f:
            api_key = f.read()[:-1]
    elif 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("Please set environmental variable OPENAI_API_KEY to your API key, or pass in --api_key_fp")
    else:
        api_key = os.environ['OPENAI_API_KEY']


    if args.run_async:
        client = AsyncOpenAI(
            api_key = api_key
        )
    else:
        client = OpenAI(
            api_key = api_key
        )

    if args.corpus_path is not None:
        logging.info("Rewriting and loading corpus from {}".format(args.corpus_path))
        corpus = dict()
        with open(args.corpus_path, "r") as f:
            for line in f:
                curr_obj = json.loads(line)
                corpus[curr_obj["id"]] = curr_obj["contents"]

    if args.corpus_path is not None:
        doc_embedding_path = args.corpus_path.replace("edit.jsonl", "doc_embeddings.npy")
        doc_ids_path = args.corpus_path.replace("edit.jsonl", "doc_ids.json")
    else:
        doc_embedding_path = os.path.join("datasets", dataset, "doc_embeddings.npy")
        doc_ids_path = os.path.join("datasets", dataset, "doc_ids.json")

    if os.path.exists(doc_embedding_path) and os.path.exists(doc_ids_path):
        documents_embeddings_np = np.load(doc_embedding_path)
        documents_embeddings = documents_embeddings_np.tolist()
        with open(doc_ids_path, "r") as f:
            doc_ids = json.load(f)
    else:
        documents_embeddings, doc_ids = get_document_embeddings(corpus, client, args)
        documents_embeddings_np = np.array(documents_embeddings)
        np.save(doc_embedding_path, documents_embeddings_np)
        with open(doc_ids_path, "w+") as f:
            json.dump(doc_ids, f)

    if args.corpus_path is not None:
        # dump embeddings in numpy
        query_embedding_path = os.path.join("datasets", dataset, "query_embeddings.npy")
        queryidx2truncatedidx_path = os.path.join("datasets", dataset, "queryidx2truncatedidx.json")
    else:
        query_embedding_path = os.path.join("datasets", dataset, "query_embeddings.npy")
        queryidx2truncatedidx_path = os.path.join("datasets", dataset, "queryidx2truncatedidx.json")

    if os.path.exists(query_embedding_path) and os.path.exists(queryidx2truncatedidx_path):
        query_embeddings_np = np.load(query_embedding_path)
        query_embeddings = query_embeddings_np.tolist()
        with open(queryidx2truncatedidx_path, "r") as f:
            queryidx2truncatedidx = json.load(f)
    else:
        query_embeddings, queryidx2truncatedidx = get_query_embeddings(queries, client, args)
        query_embeddings_np = np.array(query_embeddings)
        np.save(query_embedding_path, query_embeddings_np)
        with open(queryidx2truncatedidx_path, "w+") as f:
            json.dump(queryidx2truncatedidx, f)

    results = {}

    assert len(query_embeddings) == len(queries), f"length of query_embeddings should be {len(queries)}, now is {len(query_embeddings)}"
    for query_id in tqdm(queries):
        query_embedding = query_embeddings[queryidx2truncatedidx[query_id]]  # for each query id, found the idx of the query in embeddings
        similarities = np.dot(documents_embeddings, query_embedding)
        results[query_id] = {}
        for doc_id, score in zip(doc_ids, similarities):
            results[query_id][doc_id] = score

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    k_values=[1,3,5,10,100]
    model = DRES(models.SentenceBERT("BAAI/bge-base-en-v1.5"), batch_size=args.batch_size, corpus_chunk_size=512*9999)

    retriever = EvaluateRetrieval(model, score_function="dot")

    #### Same the original results to the dataset file
    def get_top_docs(task_id: str, topk: int = 10) -> list[str]:
        if task_id not in results:
            return []
        doc_scores = results[task_id]
        doc_scores_sorted = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        doc_scores_sorted = doc_scores_sorted[:topk]
        doc_code_snippets = [corpus[code_id] for code_id, score in doc_scores_sorted]
        return doc_code_snippets


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
                                       "docs": all_doc}) + "\n")  # didn't dump the whole data because `ans` field is inconsistent with numpy.int64 and float64 and np series
    elif args.dataset.startswith("repoeval"):
        # "../benchmarks/repoeval/datasets/api_level_completion_1k_context_codegen.test.jsonl"
        tasks = [json.loads(line.strip()) for line in open(args.dataset_path, 'r')]
        prompts, references, docs = [], [], []
        for task in tasks:
            if task["metadata"]["task_id"] not in queries:
                continue
            prompts.append(task["prompt"]) # save full prompt
            references.append(task["metadata"]["ground_truth"])
            docs.append(get_top_docs(task["metadata"]["task_id"]))
        assert len(prompts) == len(references) == len(docs), f"length of prompts, references, and docs should be the same, now is {len(prompts)}, {len(references)}, {len(docs)}"
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

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    if args.dataset != "livecodebench" and args.corpus_path is None:
        logging.info("Retriever evaluation for k in: {}".format(k_values))
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)

        mrr = retriever.evaluate_custom(qrels, results, k_values, metric="mrr")
        recall_cap = retriever.evaluate_custom(qrels, results, k_values, metric="r_cap")
        hole = retriever.evaluate_custom(qrels, results, k_values, metric="hole")

        all_results = {"ndcg": ndcg, "mrr": mrr, "recall": recall, "precision": precision}
        with open(args.output_file, "w") as f:
            json.dump(all_results, f)
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
