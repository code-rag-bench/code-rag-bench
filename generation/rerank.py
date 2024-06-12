"""Reranking top100 retrieved documents."""

import json
import torch
import argparse
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    device = torch.device("cuda")
    model.to(device)
    print("Model Device: ", model.device)
    model.eval()

    results = [json.loads(l.strip()) for l in open(args.results_path)]
    if isinstance(results[0], list):
        results = json.load(open(args.results_path))
    if args.query_field is None:
        args.query_field = input(f"Choose query field from [{results[0].keys()}]: ")

    def rerank_docs(query: str, docs: list[str]) -> list[str]:
        pairs = [[query, doc] for doc in docs]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {k: v.to(device) for k,v in inputs.items()}
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float().cpu()
        scores = np.array(scores)
        # print(scores)
        ranking = np.argsort(scores)[::-1]
        ranked_docs = [docs[i] for i in ranking]
        return ranked_docs

    reranked_results = []
    for i, r in enumerate(results):
        rr = {k:v for k,v in r.items() if k!="docs"}
        docs = r["docs"]
        if len(docs) > 0:
            assert isinstance(docs, list)
            if isinstance(docs[0], dict):
                docs = [d["text"] for d in docs]
            assert all([isinstance(d, str) for d in docs])

            ranked_docs = rerank_docs(query=r[args.query_field], docs=docs)
            rr["docs"] = [{"text": rd} for rd in ranked_docs]
        else:
            rr["docs"] = [d for d in docs]
        reranked_results.append(rr)

        if (i + 1) % args.report_steps == 0: print(f"Step #{i+1}")
    
    output_path = args.results_path.replace(".json", "_reranked.json")
    with open(output_path, 'w') as fw:
        for rr in reranked_results:
            fw.write(json.dumps(rr) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--query_field", type=str, default=None)
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--report_steps", type=int, default=100)
    args = parser.parse_args()

    main()
