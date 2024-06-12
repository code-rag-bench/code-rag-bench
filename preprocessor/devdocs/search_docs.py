"""Search Ground-Truth Library Documentation."""

import re
import ast
import argparse
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset
from detect_library import detect_library_functions

LIBRARIES = {
    'sys', 'filecmp', 'shutil', 'ast', 'functools', 'numpy', 
    'ftplib', 'pytz', 'flask', 'math', 'time', 'argparse', 
    'pandas', 'pickle', 'matplotlib', 'collections', 'queue', 
    'datetime', 'requests', 'heapq', 'socket', 'multidict', 'operator', 
    'codecs', 'imp', 'obspy', 'json', 'os', 'urllib', 'random', 'torch',
    'pprint', 're', 'sqlite3', 'struct', 'csv', 'base64', 'django', 
    'subprocess', 'regex', 'scipy', 'warnings', 'glob', 'bs4', 'sqlalchemy', 'itertools'
}

ALIAS = {
    "np": "numpy",
    "pd": "pandas",
    "re": "regex",
}

STANDARD = set(["python.library.stdtypes", "python.library.functions"])
STDLIB = {
    # text processing
    'string', 're', 'difflib', 'textwrap', 'unicodedata', 'stringprep', 'readline', 'rlcompleter',
    # binary data services
    'struct', 'codecs',
    # data types
    'datetime', 'calendar', 'collections', 'heapq', 'bisect', 'array', 'weakref', 'types', 'copy', 'pprint', 'reprlib',
    'enum',
    # numeric and mathematical modules
    'numbers', 'math', 'cmath', 'decimal', 'fractions', 'random', 'statistics',
    # functional programming modules
    'itertools', 'functools', 'operator',
    # file and directory access
    'pathlib', 'fileinput', 'stat', 'filecmp', 'tempfile', 'glob', 'fnmatch', 'linecache', 'shutil', 'macpath',
    # data persistence
    'pickle', 'copyreg', 'shelve', 'marshal', 'dbm', 'sqlite3',
    # data compression and archiving
    'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile',
    # file formats
    'csv', 'configparser', 'netrc', 'xdrlib', 'plistlib',
    # crypographic services
    'hashlib', 'hmac', 'secrets',
    # generic operating system services
    'os', 'io', 'time', 'argparse', 'getopt', 'logging', 'getpass', 'curses', 'platform', 'errno', 'ctypes',
    # concurrent execution
    'threading', 'multiprocessing', 'concurrent', 'subprocess', 'sched', 'queue', '_thread', '_dummy_thread',
    'dummy_threading',
    # contextvars
    'contextvars',
    # networking and interprocess communication
    'asyncio', 'socket', 'ssl', 'select', 'selectors', 'asyncore', 'asynchat', 'signal', 'mmap',
    # internet data handling
    'email', 'json', 'mailcap', 'mailbox', 'mimetypes', 'base64', 'binhex', 'binascii', 'quopri', 'uu',
    # structured markup processing tools
    'html', 'xml',
    # internet protocols and support
    'webbrowser', 'cgi', 'cgitb', 'wsgiref', 'urllib', 'ftplib', 'poplib', 'imaplib', 'nntplib', 'smtplib', 'smtpd',
    'telnetlib', 'uuid', 'socketserver', 'xmlrpc', 'ipaddress',
    # multimedia
    'audioop', 'aifc', 'sunau', 'wave', 'chunk', 'colorsys', 'imghdr', 'sndhdr', 'ossaudiodev',
    # internationalization
    'gettext', 'locale',
    # program frameworks
    'turtle', 'cmd', 'shlex',
    # graphical user interfaces with tk
    'tkinter',
    # development tools
    'typing', 'pydoc', 'doctest', 'unittest', 'lib2to3', 'test',
    # debugging and profiling
    'bdb', 'faulthandler', 'pdb', 'timeit', 'trace', 'tracemalloc',
    # software packaging and distribution
    'distutils', 'ensurepip', 'venv', 'zipapp',
    # python runtime services
    'sys', 'sysconfig', 'builtins', 'warnings', 'dataclasses', 'contextlib',
    'abc', 'atexit', 'traceback', '__future__', 'gc', 'inspect', 'site',
    # custom python interpreters
    'code', 'codeop',
    # importing modules
    'zipimport', 'pkgutil', 'modulefinder', 'runpy', 'importlib',
    # python language services
    'parser', 'ast', 'symtable', 'symbol', 'token', 'keyword', 'tokenize', 'tabnanny', 'pyclbr', 'py_compile',
    'compileall', 'dis', 'pickletools',
    # miscellaneous services
    'formatter',
    # ms windows specific services
    'msilib', 'msvcrt', 'winreg', 'winsound',
    # unix specific services
    'posix', 'pwd', 'spwd', 'grp', 'crypt', 'termios', 'tty', 'pty', 'fcntl', 'pipes', 'resource', 'nis', 'syslog',
    # superseded modules
    'optparse', 'imp',
    # undocumented modules
    'posixpath', 'ntpath'
}
LIB_MAP = {
    "regex": "re",
    "pytorch": "torch",
}


# Define a visitor class to traverse the AST
class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.library_functions = set()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            # If the function is an attribute, it's a method call
            library_function = node.func.attr
            if isinstance(node.func.value, ast.Call):
                if isinstance(node.func.value.func, ast.Call):
                    library_name = node.func.value.func.attr
                elif isinstance(node.func.value.func, ast.Name):
                    library_name = node.func.value.func.id
                else:
                    library_name = ""
            elif isinstance(node.func.value, ast.Name):
                library_name = node.func.value.id
            elif isinstance(node.func.value, ast.Attribute):
                if isinstance(node.func.value.value, ast.Name):
                    library_name = node.func.value.value.id
                elif isinstance(node.func.value.value, ast.Attribute):
                    library_name = node.func.value.value.value
                else:
                    library_name = ""
            else:
                library_name = ""
            if library_name in ALIAS: library_name = ALIAS[library_name]
            if library_name:
                self.library_functions.add(f"{library_name}.{library_function}")
            else:
                self.library_functions.add("library_function")
        elif isinstance(node.func, ast.Name):
            # If the function is a name, it's a regular function call
            library_function = node.func.id
            self.library_functions.add(f"{library_function}")

    def visit_Attribute(self, node):
        # Handling attribute accesses
        if isinstance(node.value, ast.Name):
            library_name = node.value.id
            if library_name in ALIAS:
                library_name = ALIAS[library_name]
            if library_name in LIBRARIES:
                library_function = node.attr
                self.library_functions.add(f"{library_name}.{library_function}")
        if isinstance(node.value, ast.Call):
            self.visit_Call(node.value)


# Function to detect library functions
def detect_library_functions(code, verbose: bool = False):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"Syntax Error: {e}")
        return []

    visitor = FunctionCallVisitor()
    visitor.visit(tree)

    if verbose:
        for func in visitor.library_functions:
            print(func)
    return visitor.library_functions

# Sample Python code
sample_code = """
import math

def calculate_area(radius):
    return math.pi * radius**2

print(calculate_area(5))
"""

# Detect library functions used in the sample code
library_functions_used = detect_library_functions(sample_code)

print("Library functions used:")
for func in library_functions_used:
    print(func)



def tokenize_python_code(code: str) -> list[str]:
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens


def check_library(doc_name_list: list[str], library: str) -> bool:
    if library in LIB_MAP:
        return (LIB_MAP[library] in doc_name_list) or (library in doc_name_list)
    elif library in STDLIB:
        return library in doc_name_list
    else:
        return doc_name_list[0] == library


def find_matched_docs(name: str | list[str], docs: list[str], libraries: list[str]) -> list[str]:
    if isinstance(name, str): name = [name]
    matched_record = {}
    for i,d in enumerate(docs):
        d_name = d.split('#')[0]
        # if no libraries used, must be standard type
        if len(libraries) == 0 and d_name not in STANDARD: continue
        # if libraries are used, must in the name path
        if len(libraries) > 0: 
            if not any([
                check_library(
                    doc_name_list=d_name.split('.'), 
                    library=lib
                )
                for lib in libraries
            ]): continue

        d_names = d.split('#')[-1].split('.')
        if not any([(n == d_names[-1]) for n in name]): continue
        count = sum([1 for n in name if n in d_names])
        if count not in matched_record:
            matched_record[count] = []
        matched_record[count].append(i)

    if len(matched_record) == 0: return []
    max_count = max(matched_record.keys())
    matched_docs = matched_record[max_count]
    return matched_docs


def get_canonical_docs(
    docs: list[str], 
    code: str, full_code: str,
    libraries: list[str],
) -> dict[str, list[int]]:
    detected_functions = detect_library_functions(code)
    if len(detected_functions) == 0:
        detected_functions = detect_library_functions(full_code.replace('\t', ' '*4))
    func_names = [n.lower().split('.') for n in detected_functions] + [n.split('.') for n in detected_functions]

    matched_docs = {}
    for fn in func_names:
        fn_matched_docs = find_matched_docs(fn, docs, libraries)
        matched_docs['.'.join(fn)] = fn_matched_docs

    return matched_docs


# Function name matching
def match_find_function_names(code: str) -> list[str]:
    tokens = tokenize_python_code(code)
    func_names = []
    for i,t in enumerate(tokens):
        if (i < len(tokens)-1) and t=='.':
            func_names.append(tokens[i+1])
    return func_names

def match_find_matched_docs(name: str, docs: list[str], libraries: list[str]) -> list[str]:
    matched_docs = []
    for i,d in enumerate(docs):
        if len(libraries) > 0 and d.split('.')[0] not in libraries:
            continue
        if len(libraries) == 0 and d.split('.')[0] != "python":
            continue
        if name in d.split('#')[-1].split('.'):
            matched_docs.append(i)
    return matched_docs

def match_get_canonical_docs(docs: list[str], code: str, libraries: list[str]) -> dict:
    func_names = match_find_function_names(code)

    matched_docs = {}
    for fn in func_names:
        fn_matched_docs = match_find_matched_docs(fn, docs, libraries)
        matched_docs[fn] = fn_matched_docs

    return matched_docs


# Truncate library documentations
def truncate_doc_content(text: str) -> str:
    lines = text.split('\n')

    # truncate when examples start
    def trim_examples(lines: list[str]) -> list[str]:
        example_index = len(lines)
        for i,l in enumerate(lines):
            if l.startswith(">>>"):
                example_index = i
                break
        return lines[: example_index]
    
    if len(lines) > 5: 
        lines = trim_examples(lines)

    # truncate when function listing start
    listing_index = len(lines)
    def is_list_function(text: str) -> bool:
        return len(text.split()) and len(text.split()[0].split('.')) == 2

    for i, l in enumerate(lines):
        if i>2 and is_list_function(l):
            listing_index = i
            break
    lines = lines[: listing_index]

    if len(lines) > 10:
        lines = lines[: 3]
    return '\n'.join(lines)


# Get silver canonical docs for datasets
def get_odex_docs():
    dataset = load_dataset("neulab/odex", "en")["test"]
    code_docs = load_dataset("neulab/docprompting-conala", "docs")["train"]

    all_docs = []
    for idx,item in enumerate(tqdm(dataset)):
        docs = []
        libraries = item["library"]
        # libraries = [lib for lib in item["library"] if lib not in STDLIB]
        if libraries:
            # visitor
            doc_index_dict = get_canonical_docs(
                docs=code_docs["doc_id"],
                code=item["canonical_solution"],
                full_code=''.join([item[k] for k in ["prompt", "canonical_solution", "suffix"]]),
                libraries=libraries
            )
            for func, indices in doc_index_dict.items():
                docs.extend([{
                    "function": func, 
                    "title": code_docs["doc_id"][ii], 
                    "text": truncate_doc_content(code_docs["doc_content"][ii])
                } for ii in indices])
            
            # text match
            match_doc_index_dict = match_get_canonical_docs(
                docs=code_docs["doc_id"],
                code=item["canonical_solution"],
                libraries=libraries
            )
            for func, indices in doc_index_dict.items():
                docs.extend([{
                    "function": func, 
                    "title": code_docs["doc_id"][ii], 
                    "text": truncate_doc_content(code_docs["doc_content"][ii])
                } for ii in indices])
        
        all_docs.append(docs)        
    
    dataset = dataset.add_column("docs", all_docs)
    dataset.to_json(args.output_path)


def get_ds1000_docs():
    dataset = load_dataset("xlangai/DS-1000")["test"]
    code_docs = load_dataset("neulab/docprompting-conala", "docs")["train"]

    all_docs = []
    for idx,item in enumerate(tqdm(dataset)):
        # visitor
        doc_index_dict = get_canonical_docs(
            docs=code_docs["doc_id"], 
            code=item["reference_code"],
            full_code=item["code_context"] + item["reference_code"],
            libraries=[item["metadata"]["library"].lower()]
        )
        docs = []
        for func, indices in doc_index_dict.items():
            docs.extend([{
                "function": func, 
                "title": code_docs["doc_id"][ii], 
                "text": truncate_doc_content(code_docs["doc_content"][ii])
            } for ii in indices])

        # text match
        match_doc_index_dict = match_get_canonical_docs(
            docs=code_docs["doc_id"],
            code=item["reference_code"],
            libraries=[item["metadata"]["library"].lower()]
        )
        for func, indices in doc_index_dict.items():
            docs.extend([{
                "function": func, 
                "title": code_docs["doc_id"][ii], 
                "text": truncate_doc_content(code_docs["doc_content"][ii])
            } for ii in indices])

        all_docs.append(docs)
    
    dataset = dataset.add_column("docs", all_docs)
    dataset.to_json(args.output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=["odex", "ds1000"])
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = f"{args.dataset_name}_docs.json"
    
    if args.dataset_name == "odex":
        get_odex_docs()
    else: # "ds1000"
        get_ds1000_docs()
