import os
import json
# import html2text

from bs4 import BeautifulSoup


html_data = {}
for (dirpath, dirnames, filenames) in os.walk('dateback'):
    for filename in filenames:
        if filename.endswith('.html'):
            full_path = os.sep.join([dirpath, filename])
            with open(full_path) as html_file:
                contents = html_file.read()
            soup = BeautifulSoup(contents, 'lxml')
            attrib = soup.find(class_='_attribution')
            if attrib:
                attrib.decompose()
            html_data[full_path.split('dateback/', maxsplit=1)[1].rsplit('.html', maxsplit=1)[0]] = soup

counter = 0
print('loaded htmls')
with open('python_list_backdate.txt') as lib_file:
    for lib_name in lib_file:
        lib_name = lib_name.strip()
        index_dict = json.load(open('dateback/' + lib_name + '/index.json'))
        populated_indexes = []
        for item in index_dict['entries']:
            html_path = lib_name + '/' + item['path'].split('#')[0]
            if html_path in html_data:
                soup = html_data[html_path]
                counter += 1
                texts = []
                if '#' in item['path']:
                    html_id = item['path'].split('#')[1]
                    # print(html_id)
                    # print(html_path)
                    func_segment = soup.find(id=html_id)
                    if func_segment:
                        paras = func_segment.parent.find_all('p')
                        for p in paras:
                            texts.append(p.get_text())
                        if not texts:
                            # empty texts
                            next_sib =  func_segment.next_sibling
                            if next_sib.name == 'p':
                                texts = [next_sib.get_text()]
                        if not texts:
                            next_sib =  func_segment.parent.next_sibling
                            if next_sib.name == 'p':
                                texts = [next_sib.get_text()]

                    else:
                        paras = soup.find_all('p')
                        for p in paras:
                            texts.append(p.get_text())

                else:
                    paras = soup.find_all('p')
                    for p in paras:
                        texts.append(p.get_text())
                                
                item['text'] = texts
                populated_indexes.append(item)
        json.dump(populated_indexes, open('dateback/' + lib_name + '/populated_index_paras.json', 'w'))
