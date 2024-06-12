import os
import html2text

text_maker = html2text.HTML2Text()
text_maker.unicode_snob = 1
text_maker.ignore_emphasis = True
text_maker.single_line_break = True
text_maker.ignore_links = True
text_maker.bypass_tables = False


for (dirpath, dirnames, filenames) in os.walk('raw'):
    for filename in filenames:
        if filename.endswith('.html'):
            full_path = os.sep.join([dirpath, filename])
            with open(full_path) as html_file:
                contents = html_file.read()
            plain_text = text_maker.handle(contents)
            
            plain_full_path = 'plain/' + full_path.split('raw/', maxsplit=1)[1]
            plain_full_path = plain_full_path.rsplit('.html', maxsplit=1)[0] + '.txt'
            # create corresponding text files
            os.makedirs(os.path.dirname(plain_full_path), exist_ok=True)
            with open(plain_full_path, 'w', encoding='utf-8') as plain_text_file:
                plain_text_file.write(plain_text)

