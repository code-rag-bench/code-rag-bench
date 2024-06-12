import argparse
import json
import os
from bs4 import BeautifulSoup
import unicodedata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='/home/velocity/code_retrieval_beir/save/',
                        help='the directory to unprocessed tutorial path')
    parser.add_argument('--output-dir', type=str, default='/home/velocity/code_retrieval_beir/processed/',
                        help='the directory to save processed tutorial path')
    args = parser.parse_args()

    files = os.listdir(args.input_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    file_paths = []
    output_paths = []
    for file in files:
        if file.endswith('.jsonl'):
            output_paths.append(os.path.join(args.output_dir, 'processed_' + file))
            file_paths.append(os.path.join(args.input_dir, file))

    for input_file, output_file in zip(file_paths, output_paths):
        print("now processing: ", input_file, " to ", output_file)
        with open(input_file, 'r') as fin:
            meta_parsed = []
            for line in fin:
                res = json.loads(line)
                parsed = BeautifulSoup(res['html'])
                uri = res['uri']

                if not parsed.find_all('div', attrs={'class': 'code-container'}) and not parsed.find_all('pre'):
                    continue

                if parsed.find('div', attrs={'class': 'article-title'}):
                    title_element = parsed.find('div', attrs={'class': 'article-title'})
                elif parsed.find('title'):
                    title_element = parsed.find('title')
                else:
                    title_element = parsed.find('h1')
                if title_element.text is None:
                    title = ''
                else:
                    title = unicodedata.normalize("NFKD", title_element.text)

                if parsed.find('div', attrs={'class': 'footer-wrapper_branding-email'}):  # geeksforgeeks
                    end_element = parsed.find('div', attrs={'class': 'footer-wrapper_branding-email'})
                elif parsed.find('footer'):  # towardsdatascience, tutorialspoint
                    end_element = parsed.find('footer')
                else:
                    end_element = None
                # start to iterate later
                parsed_results = []
                all_text = ''
                curr = title_element
                while curr != end_element:
                    if not curr:
                        break
                    if curr.name == 'p' or curr.name == 'pre' or curr.name == 'li' or curr.name == 'ol':
                        curr_text = curr.get_text()
                        if curr_text == '' or not curr_text:
                            curr = curr.next
                            continue
                        curr_text = unicodedata.normalize("NFKD", curr_text)
                        all_text += curr_text + '\n'
                        parsed_results.append({"text": curr_text})
                    elif curr.name == 'div' and 'class' in curr.attrs and 'code-container' in curr.attrs['class']:
                        curr_text = curr.get_text()
                        if curr_text == '' or not curr_text:
                            curr = curr.next
                            continue
                        curr_text = unicodedata.normalize("NFKD", curr.get_text())
                        all_text += curr_text + '\n'
                        parsed_results.append({"code": curr_text})
                    elif curr.name == 'div' and 'class' in curr.attrs and 'article-bottom-text' in curr.attrs['class']:
                        break
                    curr = curr.next
                curr_obj = {"title": title, "uri": uri, "text": all_text, "parsed": parsed_results}
                meta_parsed.append(curr_obj)

        with open(output_file, 'w+') as fout:
            for res in meta_parsed:
                fout.write(json.dumps(res))
                fout.write('\n')

if __name__ == '__main__':
    main()
