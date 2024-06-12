import json
import tqdm

with open('../redpajama/stackexchange.jsonl', 'r') as input_file, open('preprocessor/redpajama/processed/stackoverflow.jsonl', 'w') as output_file:
    for line in tqdm.tqdm(input_file, total=29825086):
        data = json.loads(line)
        if data['meta']['url'].startswith('https://stackoverflow.com'):
            output_file.write(json.dumps(data) + '\n')
