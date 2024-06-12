"""Post-process docs before generation.
- solutions: none
- tutorials: first 200 tokens, first 500 tokens, none
"""

import json
import argparse
from transformers import AutoTokenizer

def get_chunks(text: str, num_tokens: int, tokenizer: AutoTokenizer) -> list[str]:
    # aggregate content line by line
    line_tokens = [tokenizer.tokenize(line) for line in text.split('\n')]

    # chunk every `num_tokens` tokens
    token_chunks = []
    curr_tokens, curr_chunk = 0, []
    for lt in line_tokens:
        if len(lt) + curr_tokens > num_tokens:
            token_chunks.append(curr_chunk)
            curr_tokens, curr_chunk = 0, []
        else:
            curr_chunk.append(lt)
            curr_tokens += len(lt)
    if curr_tokens > 0:
        token_chunks.append(curr_chunk)
    
    # convert tokens back to text strings
    text_chunks = []
    for ck in token_chunks:
        ck = [tokenizer.convert_tokens_to_string(lt) for lt in ck]
        ck = '\n'.join(ck)
        text_chunks.append(ck)
    return text_chunks

def truncate_documentation(text: str) -> str:
    lines = text.split('\n')

    # truncate when examples start
    def trim_examples(lines: list[str]) -> list[str]:
        example_index = len(lines)
        for i,l in enumerate(lines):
            if l.startswith(">>>"):
                example_index = i
                break
        return lines[: example_index]
    
    if len(lines) > 5: 
        lines = trim_examples(lines)

    # truncate when function listing start
    listing_index = len(lines)
    def is_list_function(text: str) -> bool:
        return len(text.split()) and len(text.split()[0].split('.')) == 2

    for i, l in enumerate(lines):
        if i>2 and is_list_function(l):
            listing_index = i
            break
    lines = lines[: listing_index]

    if len(lines) > 10:
        lines = lines[: 3]
    return '\n'.join(lines)


def truncate_text(text: str, num_tokens: int, is_docs: bool, tokenizer: AutoTokenizer) -> str:
    if num_tokens is None: 
        if is_docs: 
            return truncate_documentation(text)
        else:
            return text
    
    return get_chunks(text, num_tokens, tokenizer)[0]


def main():
    results = [json.loads(l.strip()) for l in open(args.results_path, 'r')]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    proc_results = []
    for r in results:
        pr = {k:v for k,v in r.items() if k!="docs"}
        pr_docs = []
        for doc in r["docs"]:
            pd = {"title": doc.get("title", None)}
            pd["text"] = truncate_text(doc["text"], args.max_num_tokens, args.is_docs, tokenizer)
            pr_docs.append(pd)
        pr["docs"] = pr_docs
        proc_results.append(pr)
    
    if args.is_docs:
        output_path = args.results_path.replace(".json", f"_trunc.json")
    else:
        output_path = args.results_path.replace(".json", f"_{args.max_num_tokens}-tokens.json")
    print("Output Result Path: ", output_path)
    with open(output_path, 'w') as fw:
        for pr in proc_results:
            fw.write(json.dumps(pr) + '\n')
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default="bigcode/starcoder2-7b")
    parser.add_argument("--max_num_tokens", type=int, default=None)
    parser.add_argument("--is_docs", action="store_true")
    args = parser.parse_args()

    main()
