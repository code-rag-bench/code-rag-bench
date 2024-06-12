import json
import argparse
from datasets import load_dataset

START_KEYWORDS = ["Who is Who", "GFG School\nProjects", "\nSave\n", "Last Updated :"]
END_KEYWORDS = ["Related Questions & Answers"]

def remove_single_lines(text: str) -> str:
    lines = text.split('\n')
    sidx = 0
    while sidx < len(lines):
        if len(lines[sidx].strip().split()) > 2: break
        sidx += 1
    
    eidx = len(lines) - 1
    while eidx > -1:
        if len(lines[eidx].strip().split()) > 2: break
        eidx -= 1

    sidx = len('\n'.join(lines[: sidx]))
    eidx = len('\n'.join(lines[: eidx+1]))
    return sidx, eidx


def remove_start_words(text: str, start_keywords: list[str]) -> int:
    sidx = 0
    for sk in start_keywords:
        if sk in text:
            sidx = max(sidx, text.index(sk) + len(sk))
    return sidx

def remove_end_words(text: str, end_keywords: list[str]) -> int:
    eidx = len(text)
    for ek in end_keywords:
        if ek in text:
            eidx = min(eidx, text.index(ek))
    return eidx 


def clean_example(
    example: dict, 
    start_keywords: list[str], 
    end_keywords: list[str]
) -> dict:
    parsed = json.loads(example["meta"]["parsed"])
    # assign index to each item
    index = 0
    text_list = []
    for i,item in enumerate(parsed):
        content = item["text"] if "text" in item else item["code"]
        text_list.append(content)
        length = len(content) + 1
        parsed[i].update({"s": index, "e": index + length})
        index += length

    # clean text
    text = '\n'.join(text_list)
    sidx, eidx = remove_single_lines(text)
    sidx = max(sidx, remove_start_words(text, start_keywords))
    eidx = min(eidx, remove_end_words(text, end_keywords))
    text = text[sidx: eidx].strip()

    item_sidx, item_eidx = 0, len(parsed)
    for i, item in enumerate(parsed):
        if item["s"] <= sidx < item["e"]:
            item_sidx = i 
            offset = sidx - item["s"]
            parsed[i]["text"] = item["text"][offset: ]
        if item["s"] < eidx <= item["e"]:
            item_eidx = i
            offset = eidx - item["s"]
            parsed[i]["text"] = item["text"][: offset]
            break
    valid_parsed = []
    for item in parsed[item_sidx: item_eidx]:
        content = item["text"] if "text" in item else item["code"]
        if content.strip() != "":
            valid_parsed.append(item)

    return {
        "title": example["meta"]["title"],
        "text": text, 
        "parsed": valid_parsed, 
    }



def main():
    tutorials = load_dataset(args.dataset_name, cache_dir=args.cache_dir)["train"]
    cleaned_tutorials = []
    for i, ex in enumerate(tutorials):
        cleaned_ex = clean_example(ex, START_KEYWORDS, END_KEYWORDS)
        if cleaned_ex["text"] != "":
            cleaned_tutorials.append(cleaned_ex)

    with open(args.output_path, 'w') as fw:
        for cex in cleaned_tutorials:
            fw.write(json.dumps(cex) + '\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="project-x/processed_tutorials")
    parser.add_argument("--cache_dir", type=str, default="/scratch/zhiruow/data")
    parser.add_argument("--output_path", type=str, default="cleaned_tutorials.jsonl")

    args = parser.parse_args()

    main()
