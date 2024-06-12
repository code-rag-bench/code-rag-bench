from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.dense import util as utils
from utils import BEIR_DATASETS
import pathlib, os, sys
import numpy as np
import torch
import logging
import datetime
import argparse
import json
from tqdm import tqdm

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def main():
    #### /print debug information to stdout

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="scifact",
                        choices=["scifact", "humaneval", "mbpp", "apps", "code_search_net_python", "livecodebench",
                                 "odex_en", "odex_es", "odex_ja", "odex_ru", "docprompting_conala", "ds1000_all_completion", "ds1000_all_insertion",
                                 "repoeval/api/huggingface_diffusers"],
                        help="Dataset to use for evaluation")
    parser.add_argument("--model", type=str, default="text-embedding-3-small", help="Sentence-BERT model to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Sentence-BERT model to use")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--embdding_path", type=str, default=None, help="Path to encoded embeddings.")
    parser.add_argument("--output_file", type=str, default=None, required=True, help="Specify the filepath if you want to save the retrieval (evaluation) results.")
    parser.add_argument("--results_file", type=str, default=None, required=True, help="Specify the filepath if you want to save the retrieval results.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the repoeval dataset.")
    parser.add_argument("--api_key_fp", type=str, default=None, help="API key file name. There SHOULD be a newline at the end of the file.")
    parser.add_argument("--run_async", action="store_true", help="Use async calls to the API.")
    args = parser.parse_args()

    # assert args.results_file.endswith('.json'), "For HF dataset purposes: please provide a .json file for the results file."

    model = models.SentenceBERT(model_path=args.model)

    if args.dataset.startswith("repoeval"):
        assert args.dataset_path is not None, "Please provide the path to the repoeval dataset."
        assert args.dataset_path.endswith(".jsonl"), "The path to the repoeval dataset should end in .jsonl"

    dataset = args.dataset

    efficiency_results = {}
    #### Download nfcorpus.zip dataset and unzip the dataset
    if dataset in BEIR_DATASETS:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
        corpus_ids, query_ids = list(corpus), list(queries)
    else:
        corpus, queries, qrels = GenericDataLoader(data_folder=os.path.join("datasets", dataset)).load(split="test")
        corpus_ids, query_ids = list(corpus), list(queries)

    #### For benchmarking using dense models, you can take any 1M documents, as it doesnt matter which documents you chose.
    #### For simplicity, we take the first 1M documents.
    number_docs = 1000000
    reduced_corpus = [corpus[corpus_id] for corpus_id in corpus_ids[:number_docs]]

    normalize = True

    #### Pre-compute all document embeddings (with or without normalization)
    #### We do not count the time required to compute document embeddings, at inference we assume to have document embeddings in-memory.
    logging.info("Computing Document Embeddings...")
    start = datetime.datetime.now()
    if normalize:
        corpus_embs = model.encode_corpus(reduced_corpus, batch_size=args.batch_size, convert_to_tensor=True, normalize_embeddings=True)
    else:
        corpus_embs = model.encode_corpus(reduced_corpus, batch_size=args.batch_size, convert_to_tensor=True)
    end = datetime.datetime.now()
    time_taken = (end - start)
    time_taken = time_taken.total_seconds() * 1000
    logging.info("Number of documents: {}, Time: {} mil sec".format(len(corpus_embs), time_taken))
    efficiency_results["index_time (ms)"] = time_taken
    efficiency_results["index_time_per_doc (ms)"] = time_taken / len(corpus_embs[0])

    #### Saving benchmark times
    time_taken_all = {}

    # only consider the first 1k queries.
    for query_id in tqdm(query_ids[:1000]):
        query = queries[query_id]

        #### Compute query embedding and retrieve similar scores using dot-product
        start = datetime.datetime.now()
        if normalize:
            query_emb = model.encode_queries([query], batch_size=1, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        else:
            query_emb = model.encode_queries([query], batch_size=1, convert_to_tensor=True, show_progress_bar=False)

        #### Dot product for normalized embeddings is equal to cosine similarity
        sim_scores = utils.dot_score(query_emb, corpus_embs)
        sim_scores_top_k_values, sim_scores_top_k_idx = torch.topk(sim_scores, 10, dim=1, largest=True, sorted=True)
        end = datetime.datetime.now()

        #### Measuring time taken in ms (milliseconds)
        time_taken = (end - start)
        time_taken = time_taken.total_seconds() * 1000
        time_taken_all[query_id] = time_taken
        # logging.info("{}: {} {:.2f}ms".format(query_id, query, time_taken))

    time_taken = list(time_taken_all.values())
    logging.info("Average search time taken (ms): {:.2f}ms".format(sum(time_taken)/len(time_taken_all)))

    #### Measuring Index size consumed by document embeddings
    corpus_embs = corpus_embs.cpu()
    cpu_memory = sys.getsizeof(np.asarray([emb.numpy() for emb in corpus_embs]))

    logging.info("Number of documents: {}, Dim: {}".format(len(corpus_embs), len(corpus_embs[0])))
    logging.info("Index size (in MB): {:.2f}MB".format(cpu_memory*0.000001))
    efficiency_results["avg_query_time (ms)"] = sum(time_taken)/len(time_taken_all)
    efficiency_results["index_size (mb)"] = cpu_memory*0.000001

    with open(args.output_file, "w") as f:
        json.dump(efficiency_results, f)

if __name__ == "__main__":
    main()
