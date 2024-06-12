# Preprocessors

This directory contains code for preprocessing original data for online tutorials, library documentations, StackOverflow posts, and GitHub repositories, into indexable retrieval documents.


## Online Tutorials

All code under the `tutorials/` directory.

### Parsing Code-related Tutorials from ClueWeb22 Web Dump
Using `parse_clueweb.py` to parse through the 200M head sample of ClueWeb22, and find HTML pages of domain matching 'tutorialspoint.com', 'w3schools.com', 'geeksforgeeks.org', 'towardsdatascience.com'. We regard them as tutorials that may help code-related tasks.

### Process Tutorials
To process tutorials and get the parsed text/code field, run `process_tutorials.py`.

To further clean unnecessary text, run `clean_tutorials.py`.

We used the further cleaned version for all experiments in the paper.


## Library Documentation

All files under the `devdocs/` directory.

### Collect Library Documentation: DevDocs.io

To get the original raw data from `devdocs.io`, we use dumps available at this url: `https://downloads.devdocs.io/<library_path>.tar.gz` where `<library_path>` refers to the actual library identifier in DevDocs.

For example, to download this set of documents: https://devdocs.io/angular~15/, you could simply download https://downloads.devdocs.io/angular~15.tar.gz

Inside directory `devdocs`, see `get_api_dict_all.py` or `get_api_dict_paras.py` for different approaches to chunking DevDocs.io documentations.

### Annotate Canonical Library Documentation
To run the first automatic step to find canonical library documentation for open-domain datasets:
```
python search_docs.py --dataset_name odex
```


## StackOverflow Posts
We use the stackoverflow split in the RedPajama 1T dataset, under the `redpajama/` directory.

To load the entire StackOverflow set, run `preprocess_stackoverflow.py`.

To load a randomly sampled subset of SO, run `preprocess_stackoverflow_small.py`


## Github Repositories
We use the GitHub split in the RedPajama 1T dataset, under the `redpajama/` directory.
We use a filtered Python subset, using `filter_python_github.py`