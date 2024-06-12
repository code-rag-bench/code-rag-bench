from time import time
from datasets import load_dataset
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.dense import DenseRetrievalParallelExactSearch as DRPES

import logging
import pathlib, os
import random
import json

import argparse
from utils import BEIR_DATASETS
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset to use for evaluation")
    parser.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5", help="Sentence-BERT model to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for retrieval")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the repoeval dataset.")
    parser.add_argument("--output_file", type=str, required=True, help="Specify the filepath if you want to save the retrieval (evaluation) results.")
    parser.add_argument("--results_file", type=str, required=True, help="Specify the filepath if you want to save the retrieval results.")
    args = parser.parse_args()

    if args.dataset.startswith("repoeval"):
        assert args.dataset_path is not None, "Please provide the path to the repoeval dataset."

    dataset =  args.dataset
    corpus, queries, qrels = GenericDataLoader(data_folder=os.path.join("datasets", args.dataset)).load(split="test")
    
    model = DRES(models.SentenceBERT(args.model), batch_size=args.batch_size, corpus_chunk_size=512*9999)
    retriever = EvaluateRetrieval(model, score_function="dot")

    #### Retrieve dense results (format of results is identical to qrels)
    start_time = time()
    results = retriever.retrieve(corpus, queries)
    end_time = time()
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

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
        assert len(all_docs) == len(
            example_ids), f"length of all_docs should be {len(example_ids)}, now is {len(all_docs)}"
        with open(args.results_file, "w+") as fout:
            for idx, all_doc in enumerate(all_docs):
                fout.write(json.dumps({"example_id": example_id,
                                       "docs": all_doc}) + "\n")
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
            ds = load_dataset("princeton-nlp/SWE-bench_Lite")["test"]
        else:
            ds = load_dataset("princeton-nlp/SWE-bench")["test"]
            indices = [i for i,ex in enumerate(ds) if ex["instance_id"] in queries]
            ds = ds.select(indices)
        all_top_docs = []
        for instance_id in ds["instance_id"]:
            all_top_docs.append(get_top_docs(instance_id))
        ds = ds.add_column("docs", all_top_docs)
        ds.to_json(args.results_file)  # this outputs to arrow format and read as .jsonl
    else:
        with open(args.results_file, 'w+') as fw:
            for curr in results:
                fw.write(json.dumps({curr: results[curr]}) + "\n")

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    if len(qrels) == 0 or args.skip_eval is True:
        logging.info("No qrels found for this dataset.")
        return
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)


    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
    hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

    all_results = {"ndcg": ndcg, "mrr": mrr, "recall": recall, "precision": precision, "time": end_time - start_time}
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
