import os
import json
import random
import logging
import pathlib
import argparse
from time import time
from datasets import load_dataset
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.dense import DenseRetrievalParallelExactSearch as DRPES
from sentence_transformers import SentenceTransformer

import glob
import pickle
from tqdm import tqdm
import numpy as np
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def get_top_docs(results: dict, corpus: dict, task_id: str, topk: int = 10) -> list[str]:
    if task_id not in results: return []
    doc_scores = results[task_id]
    doc_scores_sorted = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    doc_scores_sorted = doc_scores_sorted[:topk]
    doc_code_snippets = [corpus[code_id] for code_id, score in doc_scores_sorted]
    return doc_code_snippets


def main():
    if os.path.exists(args.results_file):
        os.remove(args.results_file)

    model = SentenceTransformer(args.model)
    retriever = EvaluateRetrieval(model, score_function="dot")

    documents, doc_ids = [], []

    if args.hf_dataset is not None:
        corpus_data = list(load_dataset(args.hf_dataset)["train"])
        corpus_ids = []
        for idx, passage in enumerate(corpus_data):
            passage["id"] = "{0}_{1}".format(args.hf_dataset.split("/")[-1], idx)
            if "text" not in passage:
                passage["text"] = passage["doc_content"]
            corpus_ids.append( "{0}_{1}".format(args.hf_dataset.split("/")[-1], idx))
        corpus = {doc["id"]: doc for doc in corpus_data}
        
    for doc_id in corpus:
        doc = corpus[doc_id]
        documents.append(doc["text"])
        doc_ids.append(doc_id)
        
    all_embeddings = {}
    for fn in glob.glob(args.embdding_path):
        ids, embeddings = pickle.load(open(fn, "rb"))
        for id, embedding in zip(ids, embeddings):
            all_embeddings[id] = embedding
    
    documents_embeddings = []
    for doc_id in doc_ids:
        documents_embeddings.append(all_embeddings[doc_id])

    all_eval_results = []
    if args.dataset.startswith("swe-bench"):
        swebench = load_dataset("princeton-nlp/SWE-bench_Lite")["test"]
        all_top_docs = [[] for _ in swebench]

    instance_list = [i for i in os.listdir("datasets") if i.startswith(f"{args.dataset}_")]
    for ins_dir in instance_list[::-1]:
        logging.info("Instance Repo: {}".format(ins_dir))
        # load data and perform retrieval
        _, queries, qrels = GenericDataLoader(
            data_folder=os.path.join("datasets", ins_dir)
        ).load(split="test")
        query_ids = list(queries)
        logging.info(f"Instance #{ins_dir}: #{len(corpus)} corpus, #{len(queries)} queries")

        start_time = time()
        if len(queries) == 1:
            queries.update({"dummy": "dummy"})
        # results = retriever.retrieve(corpus, queries)

        results = {}
        query_embeddings = []  # this follows the order of truncated_queries, make sure that the order is correct
        # Generate embeddings in batches
        for i in tqdm(range(0, len(queries), args.batch_size)):
            end = min(len(queries), i + args.batch_size)
            batch_ids = query_ids[i:end]
            batch = [queries[q_id] for q_id in batch_ids]

            batch_embeddings = model.encode(batch, convert_to_tensor=True)

            # Add to the list of embeddings
            query_embeddings.extend(batch_embeddings)

        assert len(query_embeddings) == len(queries), f"length of query_embeddings should be {len(queries)}, now is {len(query_embeddings)}"
        for id in tqdm(range(len(queries))):
            query_id = query_ids[id]
            query_embedding = query_embeddings[id]  # for each query id, found the idx of the query in embeddings
            similarities = np.dot(documents_embeddings, query_embedding.cpu())
            results[query_id] = {}
            for doc_id, score in zip(doc_ids, similarities):
                results[query_id][str(doc_id)] = float(score)
            
        if "dummy" in queries:
            queries.pop("dummy")
            results.pop("dummy")
        end_time = time()
        logging.info("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

        # get topk retrieved docs
        if args.dataset.startswith("swe-bench"):
            indices = [i for i,ex in enumerate(swebench) if ex["instance_id"] in queries]
            for index in indices:
                instance_id = swebench[index]["instance_id"]
                all_top_docs[index] = get_top_docs(results, corpus, instance_id)
        elif args.dataset.startswith("repoeval"):
            args.dataset_path = "output/repoeval/datasets/function_level_completion_2k_context_codex.test.clean.jsonl"
            tasks = [json.loads(line.strip()) for line in open(args.dataset_path, 'r')]
            prompts, references, docs = [], [], []
            for task in tasks:
                if task["metadata"]["task_id"] not in queries: continue
                prompts.append(task["prompt"]) # save full prompt
                references.append(task["metadata"]["ground_truth"])
                docs.append(get_top_docs(
                    results=results, corpus=corpus, task_id=task["metadata"]["task_id"],
                ))
            assert len(prompts) == len(references) == len(docs)
            dataset = [
                {"prompt": p, "reference": r, "docs": d}
                for p,r,d in zip(prompts, references, docs)
            ]
            with open(args.results_file, "a") as fout:
                for curr in dataset:
                    fout.write(json.dumps(curr, default=str) + "\n")
        else:
            raise ValueError(f"`dataset` should starts with either 'swe-bench' or 'repoeval'.")

        # evaluate retrieval results
        
        if len(qrels) == 0 or args.skip_eval is True:
            logging.info("No qrels found for this dataset.")
            return
        logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
        eval_results = {
            "ndcg": ndcg, "mrr": mrr,
            "recall": recall, "precision": precision,
            # "time": end_time - start_time
        }
        logging.info(f"Instance #{ins_dir}: {eval_results}")
        all_eval_results.append(eval_results)

    if args.dataset.startswith("swe-bench"):
        swebench = swebench.add_column("docs", all_top_docs)
        swebench.to_json(args.results_file)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="swe-bench-lite",
                        choices=["swe-bench-lite", "repoeval"],
                        help="Dataset to use for evaluation")
    parser.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5", help="Sentence-BERT model to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for retrieval")
    parser.add_argument("--multi_gpu", action="store_true", help="set to use multiple GPUs for retrieval")
    parser.add_argument("--output_file", type=str, default="outputs.json",
                        help="Specify the filepath if you want to save the retrieval (evaluation) results.")
    parser.add_argument("--results_file", type=str, default="results.json",
                        help="Specify the filepath if you want to save the retrieval results.")
    parser.add_argument("--hf_dataset", type=str, help="load passages from HF dataset.")
    parser.add_argument("--embdding_path", type=str, default=None, help="Path to encoded embeddings.")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    args = parser.parse_args()

    main()
