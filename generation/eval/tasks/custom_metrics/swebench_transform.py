import json
import glob
import argparse

parser = argparse.ArgumentParser(description='Transform generation output in place for SWE-bench eval harness.')
parser.add_argument('output_path', type=str, help='path to generated model-name.json')

args = parser.parse_args()

if __name__ == '__main__':
    file_path = args.output_path
    new_data = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        for record in data:
            example = record
            if type(record) == list: 
                example = record[0]
            example['model_name_or_path'] = file_path.split('/')[-1].split('.json')[0]
            new_data.append(example)


    with open(file_path, 'w') as f:
        json.dump(new_data, f)
