import argparse
import json
import os
import subprocess

from datasets import load_dataset

def modify_single_dataset(dataset_dir, output_metadir, dataset):
    input_path = os.path.join(dataset_dir, dataset, "corpus.jsonl")
    output_dir = os.path.join(output_metadir, f"{dataset}_corpus")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "edit.jsonl")
    results = []
    with open(input_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    with open(output_path, 'w+') as f:
        for result in results:
            result['contents'] = result['text']
            result['id'] = result['_id']
            del result['_id']
            del result['text']
            f.write(json.dumps(result) + '\n')

def modify_single_dataset_repo(dataset_dir, ins_dir, output_metadir, dataset):
    output_dir = os.path.join(output_metadir, f"{dataset}_corpus")
    input_path = os.path.join(dataset_dir, ins_dir, "corpus.jsonl")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, ins_dir)):
        os.makedirs(os.path.join(output_dir, ins_dir))
    output_path = os.path.join(output_dir, ins_dir, f"edit.jsonl")
    results = []
    with open(input_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    with open(output_path, 'w+') as f:
        for result in results:
            result['contents'] = result['text']
            result['id'] = result['_id']
            del result['_id']
            del result['text']
            f.write(json.dumps(result) + '\n')


def index_single_dataset(output_metadir, index_dir, dataset):
    input_dir = os.path.join(output_metadir, f"{dataset}_corpus")
    output_dir = os.path.join(index_dir, f"{dataset}_corpus")
    subprocess.run(["python", "-m", "pyserini.index",
                    "-collection", "JsonCollection",
                    "-generator", "DefaultLuceneDocumentGenerator",
                    "-threads", "20",
                    "-input", input_dir,
                    "-index", output_dir,
                    "-storePositions", "-storeDocvectors", "-storeContents", "-pretokenized"])


def index_single_dataset_repo(output_metadir, ins_dir, index_dir, dataset):
    input_dir = os.path.join(output_metadir, f"{dataset}_corpus", ins_dir)
    output_dir = os.path.join(index_dir, f"{dataset}_corpus", ins_dir)
    subprocess.run(["python", "-m", "pyserini.index",
                    "-collection", "JsonCollection",
                    "-generator", "DefaultLuceneDocumentGenerator",
                    "-threads", "20",
                    "-input", input_dir,
                    "-index", output_dir,
                    "-storePositions", "-storeDocvectors", "-storeContents", "-pretokenized"])


def search_single_dataset(index_dir, dataset, top_k=10, k1=1.5, b=0.75):
    index_dir = os.path.join(index_dir, f"{dataset}_corpus")
    subprocess.run(["python", "eval_beir_pyserini.py", "--dataset", dataset,
                    "--index_path", index_dir,
                    "--output_file", f"results/{dataset}_k1={k1}_b={b}_pyserini_bm25_output.jsonl",
                    "--results_file", f"results/{dataset}_k1={k1}_b={b}_pyserini_bm25.jsonl",
                    "--top_k", str(top_k),
                    "--k1", str(k1),
                    "--b", str(b)])


if __name__ == '__main__':
    all_datasets = ["humaneval", "mbpp",
                    "odex_en", "ds1000_all_completion", "repoeval/function",
                    "livecodebench", "code_search_net_python"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="all",
                        choices=all_datasets+["all"])
    parser.add_argument('--dataset_dir', type=str, default='datasets')
    parser.add_argument('--output_metadir', type=str, default='/mnt/downloads')
    parser.add_argument('--index_dir', type=str, default='/mnt/downloads/bm25_indices')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--k1', type=float, default=1.2)
    parser.add_argument('--b', type=float, default=0.75)
    parser.add_argument('--stage', type=str, default='all', choices=['all','preprocess', 'index', 'search'])
    args = parser.parse_args()
    results = []

    if args.dataset == "all":
        datasets = all_datasets
    else:
        datasets = [args.dataset]

    if args.stage == 'preprocess' or args.stage == 'all':
        for dataset in datasets:
            print(f"Modifying {dataset} dataset")
            modify_single_dataset(args.dataset_dir, args.output_metadir, dataset)

    if args.stage == 'index' or args.stage == 'all':
        for dataset in datasets:
            print(f"Indexing {dataset} dataset")
            index_single_dataset(args.output_metadir, args.index_dir, dataset)

    if args.stage == "search" or args.stage == "all":
        for dataset in datasets:
            print(f"Searching {dataset} dataset")
            search_single_dataset(args.index_dir, dataset, args.top_k, args.k1, args.b)

