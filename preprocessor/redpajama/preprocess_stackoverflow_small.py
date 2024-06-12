import json
import tqdm

with open('../redpajama/small/stackexchange_sample.jsonl', 'r') as input_file, open('preprocessor/redpajama/processed/stackoverflow_small.jsonl', 'w') as output_file:
    for line in tqdm.tqdm(input_file):
        data = json.loads(line)
        if data['meta']['url'].startswith('https://stackoverflow.com'):
            output_file.write(json.dumps(data) + '\n')
