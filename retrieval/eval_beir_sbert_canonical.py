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
from tqdm import tqdm

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

    model = DRES(
        models.SentenceBERT(args.model),
        batch_size=args.batch_size,
        corpus_chunk_size=512*9999
    )
    retriever = EvaluateRetrieval(model, score_function="dot")

    if args.dataset.startswith("swe-bench") or args.dataset.startswith("repoeval"):
        all_eval_results = []
        
        if args.dataset.startswith("swe-bench"):
            swebench = load_dataset("princeton-nlp/SWE-bench_Lite")["test"]
            all_top_docs = [[] for _ in swebench]

        instance_list = [i for i in os.listdir("datasets") if i.startswith(f"{args.dataset}_")]
        instance_list_filtered = []
        
        for ins_dir in tqdm(instance_list):
            logging.info("Instance Repo: {}".format(ins_dir))
            # load data and perform retrieval
            corpus, queries, qrels = GenericDataLoader(
                data_folder=os.path.join("datasets", ins_dir)
            ).load(split="test")
            logging.info(f"Instance #{ins_dir}: #{len(corpus)} corpus, #{len(queries)} queries")

            start_time = time()
            if len(queries) == 1:
                queries.update({"dummy": "dummy"})
            results = retriever.retrieve(corpus, queries)
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
                prompts, references, docs, metadatas = [], [], [], []
                for task in tasks:
                    if task["metadata"]["task_id"] not in queries: continue
                    prompts.append(task["prompt"]) # save full prompt
                    references.append(task["metadata"]["ground_truth"])
                    docs.append(get_top_docs(
                        results=results, corpus=corpus, task_id=task["metadata"]["task_id"],
                    ))
                    metadatas.append(task["metadata"])
                assert len(prompts) == len(references) == len(docs)
                dataset = [
                    {"prompt": p, "reference": r, "docs": d, "metadata":m}
                    for p,r,d,m in zip(prompts, references, docs, metadatas)
                ]
                with open(args.results_file, "a") as fout:
                    for curr in dataset:
                        fout.write(json.dumps(curr) + "\n")
            else:
                raise ValueError(f"`dataset` should starts with either 'swe-bench' or 'repoeval'.")

            # evaluate retrieval results
            if len(qrels) == 0:
                logging.info("No qrels found for this dataset.")
                return
            logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
            eval_results = {
                "ndcg": ndcg, "mrr": mrr,
                "recall": recall, "precision": precision,
                "time": end_time - start_time
            }
            logging.info(f"Instance #{ins_dir}: {eval_results}")
            all_eval_results.append(eval_results)
            
            with open(args.output_file + "_all", "w") as f:
                json.dump(all_eval_results, f)

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
    else:
        dataset =  args.dataset
        corpus, queries, qrels = GenericDataLoader(data_folder=os.path.join("datasets", args.dataset)).load(split="test")
        #### Retrieve dense results (format of results is identical to qrels)
        start_time = time()
        results = retriever.retrieve(corpus, queries)
        end_time = time()
        print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

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
                all_top_docs.append(get_top_docs(results, corpus, f"{task_id}_doc"))
            ds["test"] = ds["test"].add_column("docs", all_top_docs)
            ds["test"].to_json(args.results_file)  # this outputs to arrow format and read as .jsonl
        elif args.dataset.startswith("odex"):
            lang = args.dataset.split("_")[-1]
            ds = load_dataset("neulab/odex", lang)
            all_top_docs = []
            for idx, task_id in enumerate(ds["test"]["task_id"]):
                all_top_docs.append(get_top_docs(results, corpus, f"{idx}_{task_id}"))
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
                all_docs.append(get_top_docs(results, corpus, example_id))
                example_ids.append(example_id)
            assert len(all_docs) == len(
                example_ids), f"length of all_docs should be {len(example_ids)}, now is {len(all_docs)}"
            with open(args.results_file, "w+") as fout:
                for idx, all_doc in enumerate(all_docs):
                    fout.write(json.dumps({"example_id": example_id,
                                        "docs": all_doc}) + "\n")
        else:
            with open(args.results_file, 'w+') as fw:
                for curr in results:
                    fw.write(json.dumps({curr: results[curr]}) + "\n")

        #### Evaluate your retrieval using NDCG@k, MAP@K ...
        if len(qrels) == 0:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="humaneval",
                        help="Dataset to use for evaluation")
    parser.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5", help="Sentence-BERT model to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for retrieval")
    parser.add_argument("--output_file", type=str, default="outputs.json",
                        help="Specify the filepath if you want to save the retrieval (evaluation) results.")
    parser.add_argument("--results_file", type=str, default="results.json",
                        help="Specify the filepath if you want to save the retrieval results.")
    args = parser.parse_args()

    main()
