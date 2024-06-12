from time import time
from datasets import load_dataset
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

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
#### /print debug information to stdout

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="scifact",
                        choices=["humaneval", "mbpp", "apps", "code_search_net", "scifact",
                                 "odex_en", "odex_es", "odex_ja", "odex_ru", "docprompting_conala"],
                        help="Dataset to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for retrieval")
    parser.add_argument("--multi_gpu", action="store_true", help="set to use multiple GPUs for retrieval")
    parser.add_argument("--initialize", action="store_true", help="Initialize or not")
    parser.add_argument("--output_file", type=str, default=None, help="Specify the filepath if you want to save the retrieval (evaluation) results.")
    parser.add_argument("--results_file", type=str, default=None, help="Specify the filepath if you want to save the retrieval results.")
    args = parser.parse_args()
    dataset =  args.dataset

    #### Download nfcorpus.zip dataset and unzip the dataset
    if dataset in BEIR_DATASETS:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)

    else:
        corpus, queries, qrels = GenericDataLoader(data_folder=os.path.join("datasets", dataset)).load(split="test")

    #### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
    # data folder would contain these files:
    # (1) nfcorpus/corpus.jsonl  (format: jsonlines)
    # (2) nfcorpus/queries.jsonl (format: jsonlines)
    # (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

    corpus, queries, qrels = GenericDataLoader(data_folder=os.path.join("datasets", args.dataset)).load(split="test")

    #### Dense Retrieval using SBERT (Sentence-BERT) ####
    #### Provide any pretrained sentence-transformers model
    #### The model was fine-tuned using cosine-similarity.
    #### Complete list - https://www.sbert.net/docs/pretrained_models.html

    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)

    retriever = EvaluateRetrieval(model)

    #### Retrieve dense results (format of results is identical to qrels)
    start_time = time()
    results = retriever.retrieve(corpus, queries)
    end_time = time()
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

    #### Same the original results to the dataset file
    def get_top_docs(task_id: str, topk: int = 10) -> list[str]:
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
        ds["test"].to_json(args.results_file)
    elif "odex" in args.dataset:
        lang = args.dataset.split("_")[-1]
        ds = load_dataset("neulab/odex", lang)
        all_top_docs = []
        for idx, task_id in enumerate(ds["test"]["task_id"]):
            all_top_docs.append(get_top_docs(f"{idx}_{task_id}"))
        ds["test"] = ds["test"].add_column("docs", all_top_docs)
        ds["test"].to_json(args.results_file)
    elif args.dataset == "docprompting_conala":
        ds = load_dataset("neulab/docprompting-conala")
        all_top_docs = []
        for idx, task_id in enumerate(ds["test"]["question_id"]):
            all_top_docs.append(get_top_docs(task_id))
        ds["test"] = ds["test"].add_column("docs", all_top_docs)
        ds["test"].to_json(args.results_file)
    else:
        with open(args.results_file, 'w+') as fw:
            for curr in results:
                fw.write(json.dumps({curr: results[curr]}) + "\n")


    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    if args.dataset != "livecodebench":
        logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

        mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
        recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
        hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

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
