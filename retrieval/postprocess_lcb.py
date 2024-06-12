import json
import os
import argparse
from beir.datasets.data_loader import GenericDataLoader


def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def get_top_docs(results: dict, corpus: dict, task_id: str, topk: int = 10) -> list[str]:
    if task_id not in results:
        return []
    doc_scores = results[task_id]
    doc_scores_sorted = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    doc_scores_sorted = doc_scores_sorted[:topk]
    doc_code_snippets = [corpus[int(code_id)] for code_id, score in doc_scores_sorted]
    return doc_code_snippets


def parse_results(results, corpus, queries, top_k=10):
    new_results = []
    for res in results:
        query_id = list(res.keys())[0]
        top_docs = get_top_docs(results=res, corpus=corpus, task_id=query_id, topk=top_k)
        new_results.append({
            "instance_id": query_id,
            "text": queries[query_id],
            "docs": top_docs
        })

    return new_results

def output_to_doc(new_results, output_path):
    with open(output_path, 'w+') as f:
        for res in new_results:
            f.write(json.dumps(res) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='livecodebench')
    parser.add_argument("--results_path", type=str, default='results/livecodebench_voyage_code.jsonl')
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    corpus, queries, qrels = GenericDataLoader(data_folder=os.path.join("datasets", args.dataset)).load(split="test")
    results = read_jsonl(args.results_path)
    new_results = parse_results(results, corpus, queries, args.top_k)
    output_to_doc(new_results, args.results_path.replace('.jsonl', '_processed.jsonl'))


if __name__ == '__main__':
    main()