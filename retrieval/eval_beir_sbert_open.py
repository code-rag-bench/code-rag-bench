from datasets import load_dataset
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import numpy as np
import tiktoken
import pickle
import glob
import asyncio
import logging
import pathlib, os
import random
import json
import argparse
import time

from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt
from sentence_transformers import SentenceTransformer

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def get_top_docs(task_id: str, topk: int = 10) -> list[str]:
    if task_id not in results:
        return []
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
    
    # Load the retrieval corpus from HF dataset
    if args.hf_dataset is not None:
        corpus_data = list(load_dataset(args.hf_dataset)["train"])
        corpus_ids = []
        for idx, passage in enumerate(corpus_data):
            passage["id"] = "{0}_{1}".format(args.hf_dataset.split("/")[-1], idx)
            if "text" not in passage:
                passage["text"] = passage["doc_content"]
            corpus_ids.append( "{0}_{1}".format(args.hf_dataset.split("/")[-1], idx))
        corpus = {doc["id"]: doc for doc in corpus_data}

    # Load pre-encoded embeddings 
    documents, doc_ids = {}, []
    for corpus_id in corpus:
        doc_ids.append(corpus_id)
        documents[corpus_id] = corpus[corpus_id]["text"]

    all_embeddings = {}
    for fn in glob.glob(args.embdding_path):
        ids, embeddings = pickle.load(open(fn, "rb"))
        for id, embedding in zip(ids, embeddings):
            all_embeddings[id] = embedding

    documents_embeddings = []
    for doc_id in doc_ids:
        documents_embeddings.append(all_embeddings[doc_id])

    if args.dataset.startswith("swe-bench") or args.dataset.startswith("repoeval"):
        all_eval_results = []
        if args.dataset.startswith("swe-bench"):
            swebench = load_dataset("princeton-nlp/SWE-bench_Lite")["test"]
            all_top_docs = [[] for _ in swebench]

        instance_list = [i for i in os.listdir("datasets") if i.startswith(f"{args.dataset}_")]
        for ins_dir in instance_list:
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
    
    else:
        # Load dataset
        dataset = args.dataset
        corpus, queries, qrels = GenericDataLoader(data_folder=os.path.join("datasets", dataset)).load(split="test")
        corpus_ids, query_ids = list(corpus), list(queries)

        # Compuete similarity
        results = {}
        query_embeddings = [] 
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
        else:
            with open(args.results_file, 'w+') as fw:
                for curr in results:
                    fw.write(json.dumps({curr:results[curr]}) + "\n")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="humaneval", help="Dataset to use for evaluation")
    parser.add_argument("--model", type=str, default="text-embedding-3-small", help="Sentence-BERT model to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Sentence-BERT model to use")
    parser.add_argument("--embdding_path", type=str, default=None, help="Path to encoded embeddings.")
    parser.add_argument("--output_file", type=str, default=None, required=True, help="Specify the filepath if you want to save the retrieval (evaluation) results.")
    parser.add_argument("--results_file", type=str, default=None, required=True, help="Specify the filepath if you want to save the retrieval results.")
    parser.add_argument("--hf_dataset", type=str, help="load passages from HF dataset.")
    args = parser.parse_args()
    
    main()
