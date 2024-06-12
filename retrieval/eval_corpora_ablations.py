import argparse
import json
import os
import subprocess

from datasets import load_dataset

def modify_single_dataset(corpus_dir, corpus, token, suffix='corpus'):
    corpus_name = corpus.split('/')[-1]
    if corpus_name == "docprompting-conala":
        corpus = load_dataset(corpus, 'docs',
                              cache_dir=os.path.join(corpus_dir, f'cache_{corpus_name}'),
                              token=token, trust_remote_code=True)['train']
    else:
        corpus = load_dataset(corpus,
                              cache_dir=os.path.join(corpus_dir, f'cache_{corpus_name}'),
                              token=token)['train']
    output_dir = os.path.join(corpus_dir, f"{corpus_name}_{suffix}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "edit.jsonl")
    with open(output_path, 'w+') as f:
        idx = 0
        for result in corpus:
            if corpus_name == "docprompting-conala":
                result['id'] = result['doc_id']
                result['contents'] = result['doc_content']
                del result['doc_id']
                del result['doc_content']
            else:
                result['contents'] = result['text']
                if 'meta' in result:
                    if 'url' in result['meta']:
                        result['id'] = result['meta']['url']
                    elif 'uri' in result['meta']:
                        result['id'] = result['meta']['uri']
                    elif 'source' in result['meta']:
                        result['id'] = corpus_name + str(idx)
                    elif 'task_id' in result['meta']:
                        result['id'] = result['meta']['task_id']
                    del result['meta']
                else:
                    result['id'] = corpus_name + str(idx)
                del result['text']
            idx += 1
            f.write(json.dumps(result) + '\n')


def index_single_dataset(corpus_dir, index_dir, corpus, suffix='corpus'):
    corpus_name = corpus.split('/')[-1]
    input_dir = os.path.join(corpus_dir, f"{corpus_name}_{suffix}")
    output_dir = os.path.join(index_dir, f"{corpus_name}_{suffix}")
    commands = ["python", "-m", "pyserini.index",
                    "-collection", "JsonCollection",
                    "-generator", "DefaultLuceneDocumentGenerator",
                    "-threads", "20",
                    "-input", input_dir,
                    "-index", output_dir,
                    "-storePositions", "-storeDocvectors", "-storeContents", "-pretokenized"]
    print(" ".join(commands))
    subprocess.run(commands)


def search_single_dataset(index_dir, corpus_path_dir, corpus, dataset, top_k=10, k1=1.2, b=0.75, suffix='corpus'):
    corpus_name = corpus.split('/')[-1]
    index_dir = os.path.join(index_dir, f"{corpus_name}_{suffix}")
    output_results_dir = os.path.join("results", f"{dataset}_k1={k1}_b={b}_pyserini_bm25_{suffix}")
    if not os.path.exists(output_results_dir):
        os.makedirs(output_results_dir)
    commands = ["python", "eval_beir_pyserini.py", "--dataset", dataset,
                    "--index_path", index_dir,
                    "--output_file", os.path.join(output_results_dir, f"{dataset}_corpus={corpus_name}_k1={k1}_b={b}_pyserini_bm25_output.jsonl"),
                    "--results_file", os.path.join(output_results_dir, f"{dataset}_corpus={corpus_name}_k1={k1}_b={b}_pyserini_bm25.jsonl"),
                    "--corpus_path", os.path.join(corpus_path_dir, f"{corpus_name}_{suffix}", "edit.jsonl"),
                    "--top_k", str(top_k),
                    "--k1", str(k1),
                    "--b", str(b)]
    print(" ".join(commands))
    subprocess.run(commands)

def search_single_dataset_repo(index_dir, corpus_path_dir, corpus, dataset, top_k=10, k1=1.2, b=0.75, suffix='corpus'):
    corpus_name = corpus.split('/')[-1]
    index_dir = os.path.join(index_dir, f"{corpus_name}_{suffix}")
    output_results_dir = os.path.join("results", f"{dataset}_k1={k1}_b={b}_pyserini_bm25_{suffix}")
    if not os.path.exists(output_results_dir):
        print(f"Creating directory {output_results_dir}")
        os.makedirs(output_results_dir)
    commands = ["python", "eval_beir_pyserini_repo.py", "--dataset", dataset,
                    "--index_path", index_dir,
                    "--output_file", os.path.join(output_results_dir, f"{dataset}_corpus={corpus_name}_k1={k1}_b={b}_pyserini_bm25_output.jsonl"),
                    "--results_file", os.path.join(output_results_dir, f"{dataset}_corpus={corpus_name}_k1={k1}_b={b}_pyserini_bm25.jsonl"),
                    "--corpus_path", os.path.join(corpus_path_dir, f"{corpus_name}_{suffix}", "edit.jsonl"),
                    "--top_k", str(top_k),
                    "--k1", str(k1),
                    "--b", str(b)]
    print(" ".join(commands))
    subprocess.run(commands)

def embed_and_search_openai_dataset(corpus_path_dir, corpus, dataset, suffix='corpus'):
    corpus_name = corpus.split('/')[-1]
    output_results_dir = os.path.join("results", f"{dataset}_openai_corpora_ablations")
    if not os.path.exists(output_results_dir):
        os.makedirs(output_results_dir)
    commands = ["python", "eval_openai.py", "--dataset", dataset,
                    "--corpus_path", os.path.join(corpus_path_dir, f"{corpus_name}_{suffix}", "edit.jsonl"),
                    "--results_file", os.path.join(output_results_dir, f"{dataset}_corpus={corpus_name}_openai.jsonl"),
                    "--output_file", os.path.join(output_results_dir, f"{dataset}_corpus={corpus_name}_openai_output.jsonl"),
                    "--api_key_fp", "../api_keys/open_ai.txt",
                    "--run_async"]
    print(commands)
    subprocess.run(commands)

def embed_and_search_openai_dataset_repo(corpus_path_dir, corpus, dataset, model, bsz, top_k, suffix='corpus'):
    corpus_name = corpus.split('/')[-1]
    output_results_dir = os.path.join("results", f"{dataset}_openai_corpora_ablations")
    if not os.path.exists(output_results_dir):
        os.makedirs(output_results_dir)
    commands = (["python", "eval_api_repo.py", "--dataset", dataset,
                    "--model", model,
                    "--batch_size", str(bsz),
                    "--top_k", str(top_k),
                    "--corpus_path", os.path.join(corpus_path_dir, f"{corpus_name}_{suffix}", "edit.jsonl"),
                    "--results_file", os.path.join(output_results_dir, f"{dataset}_corpus={corpus_name}_openai.jsonl"),
                    "--output_file", os.path.join(output_results_dir, f"{dataset}_corpus={corpus_name}_openai_output.jsonl"),
                    "--api_key_fp", "../api_keys/open_ai.txt",
                    "--run_async"])
    print(commands)
    subprocess.run(commands)

if __name__ == '__main__':
    all_datasets = ["humaneval", "odex_en"]
    all_corpora = ["project-x/programming_solutions",
                   "project-x/code-retrieval-stackoverflow-small",
                   "project-x/code-retrieval-stackoverflow",
                   "project-x/tutorials-clean",
                   "project-x/code-retrieval-github-python",
                   "project-x/code-retrieval-github-small",
                   "neulab/docprompting-conala"]
    openai_not_eval_corpora = ["project-x/code-retrieval-stackoverflow",
                               "project-x/code-retrieval-github-python"]
    repo_dataset = ["repoeval"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="humaneval",
                        choices=all_datasets+repo_dataset+['all'])
    parser.add_argument('--corpus_dir', type=str, default='/mnt/downloads/project-x-corpora')
    parser.add_argument('--model', type=str, default=['bm25', 'text-embedding-3-small'])
    parser.add_argument('--output_metadir', type=str, default='/mnt/downloads')
    parser.add_argument('--index_dir', type=str, default='/mnt/downloads/bm25_indices')
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for openAI models")
    parser.add_argument('--corpus', type=str, default='project-x/programming_solutions',
                        choices=all_corpora+['all'])
    parser.add_argument('--token_path', type=str, default='../api_keys/hf_token.txt')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--k1', type=float, default=1.2)
    parser.add_argument('--b', type=float, default=0.75)
    parser.add_argument('--stage', type=str, default='all', choices=['all', 'preprocess', 'index', 'search'])
    args = parser.parse_args()
    results = []

    with open(args.token_path) as f:
            token = f.read()[:-1]  # remove newline

    if args.corpus == "all":
        corpora = all_corpora
        if args.model != 'bm25':
            for corpus in openai_not_eval_corpora:
                corpora.remove(corpus)
    else:
        corpora = [args.corpus]

    if args.dataset == "all":
        datasets = all_datasets + repo_dataset
    else:
        datasets = [args.dataset]

    if not os.path.exists(args.corpus_dir):
        os.makedirs(args.corpus_dir)

    if args.model == 'bm25':
        if args.stage in ['preprocess', 'all']:
            for corpus in corpora:
                print(f"Modifying {corpus} corpus")
                modify_single_dataset(args.corpus_dir, corpus, token)

        if args.stage in ['index', 'all']:
            for corpus in corpora:
                print(f"Indexing {corpus} corpus")
                index_single_dataset(args.corpus_dir, args.index_dir, corpus)

        if args.stage in ['search', 'all']:
            for dataset in datasets:
                if dataset not in repo_dataset:
                    for corpus in corpora:
                        print(f"Searching {dataset} dataset on {corpus} corpus")
                        search_single_dataset(args.index_dir, args.corpus_dir, corpus, dataset, args.top_k, args.k1, args.b)
                else:
                    args.top_k = 50
                    for corpus in corpora:
                        print(f"Searching {dataset} dataset on {corpus} corpus")
                        search_single_dataset_repo(args.index_dir, args.corpus_dir, corpus, dataset, args.top_k, args.k1, args.b)

    else:
        if args.stage in ['search', 'all']:
            for dataset in datasets:
                if dataset not in repo_dataset:
                    for corpus in corpora:
                        print(f"Searching {dataset} dataset on {corpus} corpus")
                        embed_and_search_openai_dataset(args.corpus_dir, corpus, dataset)
                else:
                    args.top_k = 50
                    for corpus in corpora:
                        print(f"Searching {dataset} dataset on {corpus} corpus")
                        embed_and_search_openai_dataset_repo(args.corpus_dir, corpus, dataset, args.model, args.batch_size, args.top_k)
