from fastwarc.warc import ArchiveIterator 
from fastwarc.stream_io import GZipStream, FileStream
from multiprocessing import Pool

import tqdm
import json
import glob


filter_domains = ['tutorialspoint.com', 'w3schools.com', 'geeksforgeeks.org', 'towardsdatascience.com']

def filter_record(record):
    if record.headers['WARC-Type'] == 'response':
        uri = record.headers.get('WARC-Target-URI')
        for domain in filter_domains:
            if domain in uri:
                return True
    return False


def process_one_warc(warc_file_path):
    save_filename = warc_file_path.split("/")[-1].replace(".warc.gz", ".jsonl")
    stream = GZipStream(FileStream(warc_file_path, 'rb'))

    with open('save/' + save_filename, 'w') as f:
        for record in ArchiveIterator(stream, strict_mode=False, parse_http=False, func_filter=filter_record):
            html_uri = record.headers.get('WARC-Target-URI')
            html = record.reader.read().decode('utf-8')
            try:
                save_str = json.dumps({'uri': html_uri, 'html': html})
            except:
                continue
            f.write(save_str + '\n')
    
    stream.close()
    print(f"Finished processing {warc_file_path}")


if __name__ == "__main__":
    existing_save_files = glob.glob('save/*.jsonl')
    existing_save_ids = [x.split("/")[-1].split('.jsonl')[0] for x in existing_save_files]
    # files currently on babel
    warc_files = glob.glob('/data/datasets/clueweb22/ClueWeb22_B/html/en/en00/en00*/*.warc.gz')
    warc_files = [x for x in warc_files if x.split("/")[-1].split('.warc.gz')[0] not in existing_save_ids]
    with Pool(32) as p:    
        for _ in tqdm.tqdm(p.imap_unordered(process_one_warc, warc_files), total=len(warc_files)):
            pass
