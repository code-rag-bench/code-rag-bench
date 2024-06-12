import json
import tqdm
import glob


for filepath in tqdm.tqdm(glob.glob("../redpajama/github/*.jsonl")):
    filename = filepath.split('/')[-1]
    with open(filepath, 'r') as input_file, open(f'../redpajama/github_fix/{filename}', 'w') as output_file:
        for line in input_file:
            data = json.loads(line)
            if 'symlink_target' not in data['meta']:
                data['meta']['symlink_target'] = ''
            output_file.write(json.dumps(data) + '\n')
