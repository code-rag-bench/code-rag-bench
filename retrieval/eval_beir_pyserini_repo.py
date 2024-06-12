from time import time
from datasets import load_dataset
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import numpy as np
import subprocess

import logging
import pathlib, os
import random
import json

import argparse
from utils import BEIR_DATASETS

from tqdm import tqdm

from pyserini.search import SimpleSearcher
from pyserini.analysis import JWhiteSpaceAnalyzer
from pyserini import analysis
from modify_corpus_for_bm25 import modify_single_dataset_repo, index_single_dataset_repo


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


def search_indexes(searcher, query, k):
    hits = searcher.search(query, k=k)
    doc_ids = []
    scores = []
    for i in range(len(hits)):
        doc_ids.append(hits[i].docid)
        scores.append(hits[i].score)
    return doc_ids, scores, []


def search_indexes_custom_corpus(searcher, query, k):
    hits = searcher.search(query, k=k)
    doc_ids = []
    scores = []
    docs = []
    for i in range(len(hits)):
        docs.append(hits[i].lucene_document.get('contents'))
        doc_ids.append(hits[i].docid)
        scores.append(hits[i].score)
    return doc_ids, scores, docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="repoeval",
                        choices=["swe-bench-lite", "repoeval"],
                        help="Dataset to use for evaluation")
    parser.add_argument("--index_path", type=str, default="/mnt/downloads/bm25_indices/", help="the path to where the pre-built index should be saved")
    parser.add_argument("--output_metadir", type=str, default="/mnt/downloads/project-x-corpora", help="the metadirectory for outputing the modified corpus")
    parser.add_argument("--batch_size", type=int, default=64, help="Sentence-BERT model to use")
    parser.add_argument("--corpus_path", type=str, default=None, help="if specified, meaning that we don't use existing corpus in the dataset, and specified a processed corpus for bm25")
    parser.add_argument("--output_file", type=str, default=None, help="Specify the filepath if you want to save the retrieval (evaluation) results.")
    parser.add_argument("--results_file", type=str, default=None, help="Specify the filepath if you want to save the retrieval results.")
    parser.add_argument('--k1', default=1.2, help='the k1 in bm25')
    parser.add_argument('--b', default=0.75, help='the b in bm25')
    parser.add_argument('--top_k', default=10, help='the number of documents output')
    args = parser.parse_args()

    if type(args.k1) == str and type(args.b) == str:
        args.k1 = float(args.k1)
        args.b = float(args.b)
    if type(args.top_k) == str:
        args.top_k = int(args.top_k)


    all_eval_results = []
    if args.dataset.startswith("swe-bench"):
        swebench = load_dataset("princeton-nlp/SWE-bench_Lite")["test"]
        all_top_docs = [[] for _ in swebench]

    instance_list = [i for i in os.listdir("datasets") if i.startswith(f"{args.dataset}_")]

    for ins_dir in instance_list:
        logging.info("Instance Repo: {}".format(ins_dir))
        # load data and perform retrieval
        corpus, queries, qrels = GenericDataLoader(
            data_folder=os.path.join("datasets", ins_dir)
        ).load(split="test")

        start_time = time()
        if args.corpus_path is not None:  # overwrite corpus
            corpus = dict()
            with open(args.corpus_path, "r") as f:
                for line in f:
                    curr_obj = json.loads(line)
                    corpus[curr_obj["id"]] = curr_obj["contents"]
            documents, doc_ids = [], []
            for doc_id in corpus:
                documents.append(corpus[doc_id])
                doc_ids.append(doc_id)
            searcher = SimpleSearcher(args.index_path)
        else:
            # modify the corpus to be in the format of pyserini
            modify_single_dataset_repo(dataset_dir="datasets",
                                       ins_dir=ins_dir,
                                       output_metadir=args.output_metadir,
                                       dataset=args.dataset)
            index_single_dataset_repo(output_metadir=args.output_metadir,
                                 ins_dir=ins_dir,
                                 index_dir=args.index_path,
                                 dataset=args.dataset)

            logging.info(f"Instance #{ins_dir}: #{len(corpus)} corpus, #{len(queries)} queries")
            searcher = SimpleSearcher(os.path.join(args.index_path, f"{args.dataset}_corpus", ins_dir))

        searcher.set_bm25(args.k1, args.b)
        searcher.set_analyzer(JWhiteSpaceAnalyzer())

        results = dict()
        queryid2docs = dict()
        for query_id in tqdm(queries):
            query = queries[query_id]
            top_doc_ids, top_scores, top_docs = search_indexes_custom_corpus(searcher, query, 100)
            queryid2docs[query_id] = top_docs
            results[query_id] = {}
            for doc_id, score in zip(top_doc_ids, top_scores):
                results[query_id][doc_id] = score

        end_time = time()
        logging.info("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
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
                if task["metadata"]["task_id"] not in queries:
                    continue
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

    if args.corpus_path is None:
        if args.dataset.startswith("swe-bench"):
            swebench = swebench.add_column("docs", all_top_docs)
            swebench.to_json(args.results_file)

        avg_eval_results = {}
        for k, v_dict in all_eval_results[0].items():
            if isinstance(v_dict, dict):
                avg_v_dict = {}
                for vk, vv in v_dict.items():
                    avg_vv = sum([e[k][vk] for e in all_eval_results]) / len(all_eval_results)
                    avg_v_dict[vk] = avg_vv
                avg_eval_results.update(avg_v_dict)
            elif isinstance(v_dict, float):
                avg_v = sum([e[k] for e in all_eval_results]) / len(all_eval_results)
                avg_eval_results[k] = avg_v
            else:
                raise ValueError
        print("Average Eval Results: ", avg_eval_results)
        with open(args.output_file, "w") as f:
            json.dump(avg_eval_results, f)

if __name__ == '__main__':
    main()