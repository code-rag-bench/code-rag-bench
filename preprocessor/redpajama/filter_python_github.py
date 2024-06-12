import json
import tqdm
import glob


for filepath in tqdm.tqdm(glob.glob("../redpajama/github/*.jsonl")):
    filename = filepath.split('/')[-1]
    with open(filepath, 'r') as input_file, open(f'../redpajama/github_python/{filename}', 'w') as output_file:
        for line in input_file:
            data = json.loads(line)
            codepath = data['meta']['path']
            if codepath.endswith('.py'):
                output_file.write(line)