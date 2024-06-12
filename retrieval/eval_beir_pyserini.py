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
from gzip import GzipFile


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def search_indexes(searcher, query, k):
    hits = searcher.search(query, k=k)
    doc_ids = []
    scores = []
    for i in range(len(hits)):
        doc_ids.append(hits[i].docid)
        scores.append(hits[i].score)
    return doc_ids, scores


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
#### /print debug information to stdout
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="livecodebench",
                        choices=["humaneval", "mbpp",
                    "code_search_net_python", "odex_en",
                    "ds1000_all_completion", "livecodebench"],
                        help="Dataset to use for evaluation")
    parser.add_argument("--index_path", type=str, default="/mnt/downloads/bm25_indices/livecodebench_corpus/", help="the path to the pre-built index")
    parser.add_argument("--batch_size", type=int, default=64, help="Sentence-BERT model to use")
    parser.add_argument("--output_file", type=str, default=None, help="Specify the filepath if you want to save the retrieval (evaluation) results.")
    parser.add_argument("--results_file", type=str, default=None, help="Specify the filepath if you want to save the retrieval results.")
    parser.add_argument('--k1', default=1.2, help='the k1 in bm25')
    parser.add_argument('--b', default=0.75, help='the b in bm25')
    parser.add_argument('--corpus_path', type=str, default=None, help="if specified, meaning that we don't use existing corpus in the dataset, and specified a processed corpus for bm25")
    parser.add_argument('--top_k', default=10, help='the number of documents output')
    args = parser.parse_args()

    if type(args.k1) == str and type(args.b) == str:
        args.k1 = float(args.k1)
        args.b = float(args.b)
    if type(args.top_k) == str:
        args.top_k = int(args.top_k)
    if args.dataset == "repoeval":
        args.top_k = 50

    dataset =  args.dataset

    #### Download nfcorpus.zip dataset and unzip the dataset
    if dataset in BEIR_DATASETS:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)

    else:
        corpus, queries, qrels = GenericDataLoader(data_folder=os.path.join("datasets", dataset)).load(split="test")
    # reload corpus, define search function
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
        queryid2docs = dict()
    else:
        documents, doc_ids = [], []
        for doc_id in corpus:
            doc = corpus[doc_id]
            documents.append(doc["text"])
            doc_ids.append(doc_id)


    def get_top_docs(task_id: str, topk:int) -> list[str]:
        if task_id not in results: return []
        if type(topk) == str:
            topk = int(topk)
        doc_scores = results[task_id]
        doc_scores_sorted = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        doc_scores_sorted = doc_scores_sorted[:topk]
        doc_code_snippets = [corpus[code_id] for code_id, score in doc_scores_sorted]
        return doc_code_snippets


    searcher = SimpleSearcher(args.index_path)
    searcher.set_bm25(args.k1, args.b)
    searcher.set_analyzer(JWhiteSpaceAnalyzer())

    results = dict()

    for query_id in tqdm(queries):
        query = queries[query_id]
        if args.corpus_path is None:
            top_doc_ids, top_scores = search_indexes(searcher, query, 100)
        else:
            top_doc_ids, top_scores, top_docs = search_indexes_custom_corpus(searcher, query, 100)
            if not(len(top_doc_ids) == len(top_scores) == len(top_docs) == 100):
                print(f"{len(top_doc_ids)}, {len(top_scores)}, {len(top_docs)}")
            if len(top_doc_ids) == 0:
                print(f"query_id: {query_id}, query: {query}")
            queryid2docs[query_id] = top_docs
        results[query_id] = {}
        for doc_id, score in zip(top_doc_ids, top_scores):
            results[query_id][doc_id] = score

    if args.dataset != "livecodebench" and args.corpus_path is None:
        # for livecodebench, we don't have the gold corpora
        # for custom corpus, qrels stop making sense

        #### Evaluate your retrieval using NDCG@k, MAP@K ...
        k_values = [1, 3, 5, 10, 100]
        model = DRES(models.SentenceBERT("BAAI/bge-base-en-v1.5"), batch_size=args.batch_size,
                     corpus_chunk_size=512 * 9999)

        retriever = EvaluateRetrieval(model, score_function="dot")

        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)

        mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
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
            if args.corpus_path is not None:
                if f"{task_id}_doc" not in queryid2docs:
                    all_top_docs.append([])
                else:
                    all_top_docs.append(queryid2docs[f"{task_id}_doc"][:args.top_k])
            else:
                all_top_docs.append(get_top_docs(f"{task_id}_doc", args.top_k))
        ds["test"] = ds["test"].add_column("docs", all_top_docs)
        ds["test"].to_json(args.results_file)  # this outputs to arrow format and read as .jsonl
    elif "odex" in args.dataset:
        lang = args.dataset.split("_")[-1]
        ds = load_dataset("neulab/odex", lang)
        all_top_docs = []
        for idx, task_id in enumerate(ds["test"]["task_id"]):
            if args.corpus_path is not None:
                if f"{idx}_{task_id}" not in queryid2docs:
                    all_top_docs.append([])
                else:
                    all_top_docs.append(queryid2docs[f"{idx}_{task_id}"][:args.top_k])
            else:
                all_top_docs.append(get_top_docs(f"{idx}_{task_id}", args.top_k))
        ds["test"] = ds["test"].add_column("docs", all_top_docs)
        ds["test"].to_json(args.results_file)  # this outputs to arrow format and read as .jsonl
    elif args.dataset == "docprompting_conala":
        ds = load_dataset("neulab/docprompting-conala")
        all_top_docs = []
        for idx, task_id in enumerate(ds["test"]["question_id"]):
            all_top_docs.append(get_top_docs(task_id, args.top_k))
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
            all_docs.append(get_top_docs(example_id, args.top_k))
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
            docs.append(get_top_docs(task["metadata"]["task_id"], args.top_k))
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
            all_top_docs.append(get_top_docs(instance_id, args.top_k))
        ds["test"] = ds["test"].add_column("docs", all_top_docs)
        ds["test"].to_json(args.results_file)  # this outputs to arrow format and read as .jsonl
    else:
        with open(args.results_file, 'w+') as fw:
            for curr in results:
                fw.write(json.dumps({curr: results[curr]}) + "\n")


if __name__ == '__main__':
    main()