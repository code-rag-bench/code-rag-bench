from datasets import load_dataset
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import numpy as np
import tiktoken
import argparse
import asyncio
import voyageai
import numpy as np
import asyncio
import logging
import pathlib, os
import random
import math
import json
from tqdm import tqdm
from time import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from openai import OpenAI, AsyncOpenAI
from eval_openai import get_document_embeddings, get_query_embeddings
from eval_voyage import get_voyage_document_embeddings, get_voyage_query_embeddings

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
openai_embedding_encoding = "cl100k_base"


def get_top_docs(results: dict, corpus: dict, task_id: str, topk: int = 10) -> list[str]:
    if task_id not in results: return []
    doc_scores = results[task_id]
    doc_scores_sorted = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    doc_scores_sorted = doc_scores_sorted[:topk]
    doc_code_snippets = [corpus[code_id] for code_id, score in doc_scores_sorted]
    return doc_code_snippets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="text-embedding-3-small", help="Sentence-BERT model to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Sentence-BERT model to use")
    parser.add_argument("--output_file", type=str, default=None, required=True,
                        help="Specify the filepath if you want to save the retrieval (evaluation) results.")
    parser.add_argument("--results_file", type=str, default=None, required=True,
                        help="Specify the filepath if you want to save the retrieval results.")
    parser.add_argument("--corpus_path", type=str, default=None,
                        help="if specified, meaning that we don't use existing corpus in the dataset, and specified a processed corpus for bm25")
    parser.add_argument("--dataset", type=str, default=None, choices=["swe-bench-lite", "repoeval"])
    parser.add_argument("--top_k", type=int, default=10, help="Number of top documents to retrieve")
    parser.add_argument("--api_key_fp", type=str, default=None,
                        help="API key file name. There SHOULD be a newline at the end of the file.")
    parser.add_argument("--run_async", action="store_true", help="Use async calls to the API.")
    args = parser.parse_args()

    if type(args.batch_size) == str:
        args.batch_size = int(args.batch_size)

    if type(args.top_k) == str:
        args.top_k = int(args.top_k)

    # load api keys
    if args.model in ["text-embedding-3-small", "text-embedding-ada-002"]:
        if args.api_key_fp is not None:
            with open(args.api_key_fp) as f:
                api_key = f.read()[:-1]
        elif 'OPENAI_API_KEY' not in os.environ:
            raise ValueError(
                "Please set environmental variable OPENAI_API_KEY to your API key, or pass in --api_key_fp")
        else:
            api_key = os.environ['OPENAI_API_KEY']
    elif args.model in ["voyage-code-2", "voyage-large-2-instruct"]:
        if args.api_key_fp is not None:
            with open(args.api_key_fp) as f:
                api_key = f.read()[:-1]
        elif 'VOYAGE_API_KEY' not in os.environ:
            raise ValueError(
                "Please set environmental variable VOYAGE_API_KEY to your API key, or pass in --api_key_fp")
        else:
            api_key = os.environ['VOYAGE_API_KEY']
    else:
        raise ValueError(f"Model {args.model} not supported.")

    all_eval_results = []
    if args.dataset.startswith("swe-bench"):
        swebench = load_dataset("princeton-nlp/SWE-bench_Lite")["test"]
        all_top_docs = [[] for _ in swebench]

    # set retriever model
    if args.model.startswith("voyage"):
        if args.run_async:
            retriever = voyageai.AsyncClient(api_key=api_key)
        else:
            retriever = voyageai.Client(api_key=api_key)
    else:
        if args.run_async:
            retriever = AsyncOpenAI(api_key=api_key)
        else:
            retriever = OpenAI(api_key=api_key)

    instance_list = [i for i in os.listdir("datasets") if i.startswith(f"{args.dataset}_")]
    for ins_dir in instance_list:
        logging.info("Instance Repo: {}".format(ins_dir))
        # load data and perform retrieval
        corpus, queries, qrels = GenericDataLoader(
            data_folder=os.path.join("datasets", ins_dir)
        ).load(split="test")
        logging.info(f"Instance #{ins_dir}: #{len(corpus)} corpus, #{len(queries)} queries")

        if args.corpus_path is not None:
            corpus = dict()
            with open(args.corpus_path, "r") as f:
                for line in f:
                    curr_obj = json.loads(line)
                    corpus[curr_obj["id"]] = curr_obj["contents"]
            documents, doc_ids = [], []
            for doc_id in corpus:
                documents.append(corpus[doc_id])
                doc_ids.append(doc_id)

        start_time = time()

        # query embedding is the same for all corpus
        if args.model.startswith("voyage"):
            query_embedding_path = os.path.join("datasets", args.dataset, ins_dir, "voyage_query_embeddings.npy")
            queryidx2sortedidx_path = os.path.join("datasets", args.dataset, ins_dir, "voyage_queryidx2truncatedidx.json")
            query_emb_func = get_voyage_query_embeddings
        elif args.model.startswith("text-embedding"):
            query_embedding_path = os.path.join("datasets", args.dataset, ins_dir, "query_embeddings.npy")
            queryidx2sortedidx_path = os.path.join("datasets", args.dataset, ins_dir, "queryidx2truncatedidx.json")
            query_emb_func = get_query_embeddings

        if not os.path.exists(os.path.join("datasets", args.dataset, ins_dir)):
            os.makedirs(os.path.join("datasets", args.dataset, ins_dir))
        if os.path.exists(query_embedding_path) and os.path.exists(queryidx2sortedidx_path):
            query_embeddings_np = np.load(query_embedding_path)
            query_embeddings = query_embeddings_np.tolist()
            with open(queryidx2sortedidx_path, "r") as f:
                queryidx2sortedidx = json.load(f)
        else:
            query_embeddings, queryidx2sortedidx = query_emb_func(queries, retriever, args)
            query_embeddings_np = np.array(query_embeddings)
            np.save(query_embedding_path, query_embeddings_np)
            with open(queryidx2sortedidx_path, "w+") as f:
                json.dump(queryidx2sortedidx, f)

        if args.corpus_path is not None:  # get document embeddings for custom corpora
            if args.model.startswith("voyage"):
                doc_embedding_path = args.corpus_path.replace("edit.jsonl", "voyage_doc_embeddings.npy")
                doc_ids_path = args.corpus_path.replace("edit.jsonl", "voyage_doc_ids.json")
                doc_id_func = get_voyage_document_embeddings
            else:
                doc_embedding_path = args.corpus_path.replace("edit.jsonl", "doc_embeddings.npy")
                doc_ids_path = args.corpus_path.replace("edit.jsonl", "doc_ids.json")
                doc_id_func = get_document_embeddings
        else:  # get document embeddings for each sub-repo dataset
            if args.model.startswith("voyage"):
                doc_embedding_path = os.path.join("datasets", args.dataset, ins_dir, "voyage_document_embeddings.npy")
                doc_ids_path = os.path.join("datasets", args.dataset, ins_dir, "voyage_doc_ids.json")
                doc_id_func = get_voyage_document_embeddings
            else:
                doc_embedding_path = os.path.join("datasets", args.dataset, ins_dir, "document_embeddings.npy")
                doc_ids_path = os.path.join("datasets", args.dataset, ins_dir, "doc_ids.json")
                doc_id_func = get_document_embeddings

        if os.path.exists(doc_embedding_path) and os.path.exists(doc_ids_path):
            documents_embeddings_np = np.load(doc_embedding_path)
            documents_embeddings = documents_embeddings_np.tolist()
            with open(doc_ids_path, "r") as f:
                doc_ids = json.load(f)
        else:
            documents_embeddings, doc_ids = doc_id_func(corpus, retriever, args)
            np.save(doc_embedding_path, documents_embeddings)
            with open(doc_ids_path, "w") as f:
                json.dump(doc_ids, f)

        assert len(query_embeddings) == len(
            queries), f"length of query_embeddings should be {len(queries)}, now is {len(query_embeddings)}"

        results = {}
        for query_id in tqdm(queries):
            query_embedding = query_embeddings[
                queryidx2sortedidx[query_id]]  # for each query id, found the idx of the query in embeddings
            similarities = np.dot(documents_embeddings, query_embedding)
            results[query_id] = {}
            for doc_id, score in zip(doc_ids, similarities):
                results[query_id][doc_id] = score

        end_time = time()
        logging.info("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
        # get topk retrieved docs
        if args.dataset.startswith("swe-bench"):
            indices = [i for i, ex in enumerate(swebench) if ex["instance_id"] in queries]
            for index in indices:
                instance_id = swebench[index]["instance_id"]
                all_top_docs[index] = get_top_docs(results, corpus, instance_id, args.top_k)
        elif args.dataset.startswith("repoeval"):
            args.dataset_path = "output/repoeval/datasets/function_level_completion_2k_context_codex.test.clean.jsonl"
            tasks = [json.loads(line.strip()) for line in open(args.dataset_path, 'r')]
            prompts, references, docs = [], [], []
            for task in tasks:
                if task["metadata"]["task_id"] not in queries: continue
                prompts.append(task["prompt"])  # save full prompt
                references.append(task["metadata"]["ground_truth"])
                docs.append(get_top_docs(
                    results=results, corpus=corpus, task_id=task["metadata"]["task_id"], topk=args.top_k
                ))
            assert len(prompts) == len(references) == len(docs)
            dataset = [
                {"prompt": p, "reference": r, "docs": d}
                for p, r, d in zip(prompts, references, docs)
            ]
            with open(args.results_file, "a") as fout:
                for curr in dataset:
                    fout.write(json.dumps(curr) + "\n")
        else:
            raise ValueError(f"`dataset` should starts with either 'swe-bench' or 'repoeval'.")

        if args.corpus_path is None:
            # evaluate retrieval results
            k_values = [1, 3, 5, 10, 100]
            # evaluate retrieval results
            logging.info("Retriever evaluation for k in: {}".format(k_values))
            evaluate_model = DRES(models.SentenceBERT("BAAI/bge-base-en-v1.5"), batch_size=args.batch_size,
                         corpus_chunk_size=512 * 9999)
            evaluate_retriever = EvaluateRetrieval(evaluate_model, score_function="dot")

            ndcg, _map, recall, precision = evaluate_retriever.evaluate(qrels, results, k_values)
            mrr = evaluate_retriever.evaluate_custom(qrels, results, k_values, metric="mrr")
            recall_cap = evaluate_retriever.evaluate_custom(qrels, results, k_values, metric="r_cap")
            hole = evaluate_retriever.evaluate_custom(qrels, results, k_values, metric="hole")
            eval_results = {
                "ndcg": ndcg, "mrr": mrr,
                "recall": recall, "precision": precision,
                "time": end_time - start_time
            }
            logging.info(f"Instance #{ins_dir}: {eval_results}")
            all_eval_results.append(eval_results)

        if args.dataset.startswith("swe-bench"):
            swebench = swebench.add_column("docs", all_top_docs)
            swebench.to_json(args.results_file)

    if args.corpus_path is None:
        avg_eval_results = {}
        for k,v_dict in all_eval_results[0].items():
            if isinstance(v_dict, dict):
                avg_v_dict = {}
                for vk,vv in v_dict.items():
                    avg_vv = sum([e[k][vk] for e in all_eval_results])/len(all_eval_results)
                    avg_v_dict[vk] = avg_vv
                avg_eval_results.update(avg_v_dict)
            elif isinstance(v_dict, float):
                avg_v = sum([e[k] for e in all_eval_results])/len(all_eval_results)
                avg_eval_results[k] = avg_v
            else:
                raise ValueError
        print("Average Eval Results: ", avg_eval_results)
        with open(args.output_file, "w") as f:
            json.dump(avg_eval_results, f)


if __name__ == "__main__":
    main()