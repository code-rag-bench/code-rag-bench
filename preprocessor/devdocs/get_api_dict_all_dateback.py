import os
import json
import html2text

from bs4 import BeautifulSoup


text_maker = html2text.HTML2Text()
text_maker.unicode_snob = 1
text_maker.ignore_emphasis = True
text_maker.single_line_break = True
text_maker.ignore_links = True
text_maker.ignore_images = True
text_maker.bypass_tables = False

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
                if '#' in item['path']:
                    html_id = item['path'].split('#')[1]
                    # print(html_path)
                    # print(html_id)
                    id_element = soup.find(id=html_id)
                    if id_element:
                        text = id_element.parent.get_text()
                    else:
                        text = soup.get_text()

                else:
                    # print(soup)
                    text = soup.get_text()

                # print(text)
                # print('=============================')
                item['text'] = text
                populated_indexes.append(item)
        json.dump(populated_indexes, open('dateback/' + lib_name + '/populated_index_all.json', 'w'))
