# CodeRAG-Bench

This is the code repository for the project ["CodeRAG-Bench: Can Retrieval Augment Code Generation?"](https://code-rag-bench.github.io/).

## Installation

Create a new environment:
```
conda env create -n crag python=3.10 -y
conda activate crag
```
And install the necessary libraries:
```
pip install -r requirements.txt
```

## Organization
- [Retrieval](retrieval/): Code to run retrieval, with BM25, dense retrievers via [sentence-transformers](https://www.sbert.net/), and proprietary API embeddings.
- [Generation](generation/): Code to run model generation and execution-based evaluation.
- [Preprocess](preprocessor/): code to preprocess raw data for retrieval pool construction, see inside the directory for details.

## Retrieval

```
cd retrieval/
```
### Dataset Preprocessing
Before running retrieval on a dataset, you need to create the datastore for it. Following
```
python -m create/${data_name}.py
# choices for ${data_name}
# basic programming: 'humaneval', 'mbpp', 'live_code_bench'
# open-domain: 'ds1000', 'odex'
# repository-level: 'repoeval_repo', 'swebench_repo'
```

#### Adding new datasets
To run a new dataset, you can simply reformat your data into the BEIR official format, which creates a dataset directory containing the three following files, `corpus.jsonl`, `queries.jsonl` and `qrel/test.tsv`.
See [code_search_net.py](benchmarks/code_search_net.py) for some example script to combert code search net to document-to-code retrieval tasks.
Make sure your resulting dataset is under `retrieval/datasets`.


### Run dense embedding models

#### Retrieval using canonical retrieval source
Run your embedding models by loading embedding models from `sentence-transformers` as follows:

```sh
python3 eval_beir_sbert_canonical.py \
    --model YOUR_MODEL_NAME_OR_PATH \
    --dataset TASK_NAME \
    --output_file PATH_TO_YOUR_SCORE_FILE \
    --results_file PATH_TO_YOUR_RETRIEVAL_RESULTS_FILE
```
By specifying the output file name `--output_file`, you can save the retrieval results as a json file.

```json
{'ndcg': {'NDCG@1': 0.61667, 'NDCG@3': 0.68203, 'NDCG@5': 0.70804, 'NDCG@10': 0.72701, 'NDCG@100': 0.74926, 'NDCG@1000': 0.75551}, 'mrr': {'MRR@1': 0.61667, 'MRR@3': 0.67278, 'MRR@5': 0.68611, 'MRR@10': 0.69368, 'MRR@100': 0.69721, 'MRR@1000': 0.69744}, 'recall': {'Recall@1': 0.58817, 'Recall@3': 0.728, 'Recall@5': 0.79294, 'Recall@10': 0.84789, 'Recall@100': 0.95, 'Recall@1000': 0.99667}, 'precision': {'P@1': 0.61667, 'P@3': 0.26444, 'P@5': 0.176, 'P@10': 0.09533, 'P@100': 0.01077, 'P@1000': 0.00113}}
```

`--results_file` indicates the file name to store retrieval results, which will be used in the subsequent RAG evaluations.


#### Generate embeddings for open retrieval

For open retrieval, you can load a corpus file on our huggingface space ([list of the corpora](https://huggingface.co/collections/code-rag-bench/retrieval-sources-665c8ad60d71762b86506a58)), and generate embeddings using single or multiple GPUs.

- Generate embeddings using a single GPU

```
python generate_embeddings.py \
    --model YOUR_MODEL_NAME_OR_PATH \
    --output_dir OUTPUT_EMBEDDING_DIR \
    --hf_datasets HF_DATASET_NAME \
    --shard_id 0 \
    --num_shards 1
```

- Generate embeddings using multiple GPUS (e.g. 8)

```
for i in {0..7}; do
  export CUDA_VISIBLE_DEVICES=${i}
  nohup python generate_embeddings.py --model_name_or_path YOUR_MODEL_NAME_OR_PATH \
  --output_dir OUTPUT_EMBEDDING_DIR \
  --hf_datasets HF_DATASET_NAME \
  --shard_id ${i} --num_shards 8 > ./log/embeddings_logs.${i} 2>&1 &
```

#### Open retrieval using generated embeddings

Now you can load generated embeddings to run open retrieval for a target dataset.

```
python3 eval_beir_sbert_open.py \
    --model avsolatorio/GIST-large-Embedding-v0 \
    --embdding_path "OUTPUT_EMBEDDING_DIR/*" \
    --dataset DATASET_NAME \
    --hf_dataset HF_DATASET_NAME \
    --output_file PATH_TO_YOUR_SCORE_FILE \
    --results_file PATH_TO_YOUR_RETRIEVAL_RESULTS_FILE
```
### Run BM25 
Start with a fresh environment with python 3.10
```sh
# install pyserini
pip install pyserini==0.25.0
# install openjdk-11 and maven (if you don't have any)
conda install -c conda-forge openjdk=11 maven -y
```
For more information of installing pyserini, please refer to [installation guide for pyserini](https://github.com/castorini/pyserini/blob/master/docs/installation.md)

#### For canonical retrieval source (non-repo)
We provide a convenient meta script to navigate your experiments:
Preprocess all the corpus file of existing datasets into pyserini indexable format. For each dataset, the modified corpus will be saved in `OUTPUT_DIR/{DATASET_NAME}_corpus/edit.jsonl`:
```sh
python3 modify_corpus_for_bm25.py \
  --dataset DATASET_NAME, "all" if you want to do operation all datasets \
  --output_metadir OUTPUT_DIR \
  --stage preprocess
```

Indexing the corpus from `OUTPUT_DIR/{DATASET_NAME}_corpus/edit.jsonl`, and the index will be saved in `INDEX_DIR/{DATASET_NAME}_corpus/`:
```sh
python3 modify_corpus_for_bm25.py \
  --dataset DATASET_NAME, "all" if you want to do operation all datasets \
  --output_metadir OUTPUT_DIR \
  --index_dir INDEX_DIR \
  --stage index
```

Search the query from the target dataset using BM25:
```sh
python3 modify_corpus_for_bm25.py \
  --dataset DATASET_NAME, "all" if you want to do operation all datasets \
  --output_metadir OUTPUT_DIR \
  --index_dir INDEX_DIR \
  --top_k TOP_K \
  --k1 K1 \
  --b B \
  --stage search
```
The score file will be saved in `results/{DATASET_NAME}_k1={K1}_b={B}_pyserini_bm25_output.jsonl`, retrieval results in `results/{DATASET_NAME}_k1={K1}_b={B}_pyserini_bm25.jsonl`

For your convenience, you can run all the stages at once by passing `--stage all` and the corresponding parameters.

#### For canonical retrieval source (repo)
We modify and index and search each instance's corpus,
```sh
python eval_beir_pyserini_repo.py \
  --dataset DATASET_NAME \
  --output_metadir OUTPUT_DIR \
  --index_dir INDEX_DIR \
  --top_k TOP_K \
  --k1 K1 \
  --b B \
  --output_file PATH_TO_YOUR_SCORE_FILE \
  --results_file PATH_TO_YOUR_RETRIEVAL_RESULTS_FILE
```
The modified corpus will be saved in `OUTPUT_DIR/{DATASET_NAME}_corpus/{instance dir}/edit.jsonl`, 
the index will be saved in `INDEX_DIR/{DATASET_NAME}_corpus/{instance dir}`. 

#### For open retrieval 
Datasets other than swe-bench-lite are supported in this meta script:
```sh
python3 eval_corpora_ablations.py \
  --model bm25 \
  --dataset DATASET_NAME, "all" if you want to do operation all datasets \
  --corpus CORPUS_NAME, "all" if you want to do operation all corpora \
  --output_metadir OUTPUT_DIR \
  --index_dir INDEX_DIR \
  --top_k TOP_K \
  --k1 K1 \
  --b B \
  --stage {preprocess, index, search, all}
```
The modified corpus will be saved in `OUTPUT_DIR/{CORPUS NAME}_corpus/edit.jsonl`, 
the index will be saved in `INDEX_DIR/{CORPUS NAME}_corpus/`. The search result will be saved in `results/{DATASET_NAME}_k1={K1}_b={B}_pyserini_bm25_corpus`, 
and the score file will be saved in `{DATASET_NAME}_corpus={CORPUS_NAME}_k1={K1}_b={B}_pyserini_bm25_output.jsonl`, retrieval results in `{DATASET_NAME}_corpus={CORPUS_NAME}_k1={K1}_b={B}_pyserini_bm25.jsonl`.

### Run API-based models

#### For canonical retrieval source
* For non-repository level datasets:
Run your API-based models by loading embeddings from the proprietary APIs as follows:
    
    ```sh
    # voyage.ai
    python3 eval_voyage.py \
        --dataset TASK_NAME \
        --model MODEL_NAME (default is voyage-code-2) \
        --api_key_fp PATH_TO_YOUR_API_KEY_FILE (need to have a new line) \
        --batch_size YOUR_BATCH_SIZE \
        --output_file PATH_TO_YOUR_SCORE_FILE \
        --results_file PATH_TO_YOUR_RETRIEVAL_RESULTS_FILE
        
    # openai
    python3 eval_openai.py \
        --dataset TASK_NAME \
        --model MODEL_NAME (default is text-embedding-3-small) \
        --api_key_fp PATH_TO_YOUR_API_KEY_FILE (need to have a new line) \
        --batch_size YOUR_BATCH_SIZE \
        --output_file PATH_TO_YOUR_SCORE_FILE \
        --results_file PATH_TO_YOUR_RETRIEVAL_RESULTS_FILE
    ```
    `--run_async` can be used to run the retrieval asynchronously.
    
    The default behavior is to cache and use the generated document embedding and doc ids of voyage in `datasets/{dataset}/voyage_doc_embeddings.npy` and `datasets/{dataset}/voyage_doc_ids.json`,
    and of openai in `datasets/{dataset}/doc_embeddings.npy`, `datasets/{dataset}/doc_ids.json`. 
    
    Query embeddings and query to ids are also cached in `datasets/{dataset}/voyage_query_embeddings.npy` and `datasets/{dataset}/voyage_queryidx2truncatedidx.json`,
    and of openai in `datasets/{dataset}/query_embeddings.npy`, `datasets/{dataset}/queryidx2truncatedidx.json`.
    Please erase them if you would like a fresh start.
* For repository level datasets:
    ```sh
    python3 eval_api_repo.py \
        --dataset TASK_NAME \
        --model MODEL_NAME \
        --api_key_fp PATH_TO_YOUR_API_KEY_FILE (need to have a new line) \
        --batch_size YOUR_BATCH_SIZE \
        --output_file PATH_TO_YOUR_SCORE_FILE \
        --results_file PATH_TO_YOUR_RETRIEVAL_RESULTS_FILE
    ```
    `--run_async` can be used to run the retrieval asynchronously. Both query embeddings, query to id and document embeddings and document ids will be saved in `datasets/{dataset}/{instance dir}`, where instance dir is one specific instance of the dataset.
  
#### For open retrieval
Datasets other than swe-bench-lite are supported.
Add `--corpus_path THE_PATH_TO_YOUR_CORPUS_FILE (we expect an edit.jsonl in a separate directory, you can use the processed one for BM25)` to the above commands. 
Query embeddings and query to ids will be cached in the same manner as above, while the document will be cached in generated document embedding and doc ids of voyage in
`THE_PARENT_DIR_OF_CORPUS_PATH/voyage_doc_embeddings.npy` and `THE_PARENT_DIR_OF_CORPUS_PATH/voyage_doc_ids.json`,
and of openai in `THE_PARENT_DIR_OF_CORPUS_PATH/doc_embeddings.npy`, `THE_PARENT_DIR_OF_CORPUS_PATH/doc_ids.json`. 
Please erase them if you would like a fresh start.

## Generation

The `main.py` script supports running code generation with any models supported by huggingface or OpenAI.

### Baseline Generation
To run no-retrieval generation on the orignal dataset, specify its huggingface dataset name in the `dataset_path` argument:
```bash
python main.py --task "humaneval" \
--model "bigcode/starcoder2-7b" \
--dataset_path "openai_humaneval" \
--allow_code_execution
```
Set `--allow_code_execution` to evaluate generations with code execution, this is required for all tasks.

Note that the `task` should align with the `dataset_path`. All tasks available are:
- basic programming: 'humaneval', 'mbpp', 'lcb' (for livecodebench)
- open domain: 'ds1000-all-completion', 'odex-en'
- repository level: 'repoeval-function', 'swebench-lite'

### Retrieval-Augmented Code Generation
Running generation with previous retrieval results, e.g., "retrieval/humaneval/gist_large.json", specify the files as follows:
```bash
python main.py --task "humaneval" \
--model "bigcode/starcoder2-7b" \
--dataset_path "json" --data_files_test "retrieval/humaneval/gist_large.json" \
--allow_code_execution
```

Running the `main.py` script will automatically conduct execution-based evaluation after the generation is finished.
However, for RepoEval(-function) and SWE-bench(-Lite) datasets, additional setups are required due to their problem complexity.

### Execution-based Evaluation

#### RepoEval(-function)

After downloading the repositories for RepoEval (e.g., under `retrieval/output/repoeval/repositories/function_level/`), obtaining retrieval results (e.g., `/path/to/retrieval/results/retriever-name.jsonl`), and obtaining the code generation outputs in previous steps (e.g., `/path/to/generation/outputs/model-name.json`), we can run execution-based evaluation.

First, build an environment called `repoeval`. Note that you can use a separate environment to run the experiments. It will automatically switch to the `repoeval` environment when running execution.

```bash
cd generation
conda env create --file eval/tasks/custom_metrics/repoeval_environment.yml -n repoeval
```

Then, run the following command to test the environment. It runs the tests under all the repositories and checks whether the original code (i.e., the ground truth code) can pass all the test cases.

```bash
cd generation
PYTHONPATH=./ python eval/tasks/custom_metrics/repoeval_execution.py
```

If all the tests are passed, you can evaluate your code generation outputs with Pass@1.

```bash
cd generation

MODEL_NAME="model-name"
RETRIEVAL_FILE="/path/to/retrieval/results/retriever-name.jsonl"
GENERATION_OUTPUTS="/path/to/generation/outputs/model-name.json"
PYTHONPATH=./ python main.py --task "repoeval-function" --model $MODEL_NAME --dataset_path "json" \
    --data_files_test $RETRIEVAL_FILE \
    --metric_output_path results/repoeval-function_${PROMPT_NAME}_4k_${MODEL_SAVE_NAME}_evaluation_results.vllm.json \
    --max_length_input 3596 --max_length_generation 4096 --precision auto \
    --save_every_k_tasks 100 --ignore_eos --model_backend vllm --new_tokens_only --topk_docs 5 \
    --allow_code_execution \
    --load_generations_path $GENERATION_OUTPUTS
```

#### SWE-bench(-Lite)

After obtaining the generation output file from previous steps, e.g. `/path/to/generation/outputs/model-name.json`

Run the following to transform the generation output file in-place for SWE-Bench evaluation harness:

```bash
python generation/eval/tasks/custom_metrics/swebench_transform.py \
--output_path /path/to/generation/outputs/model-name.json
```

Then, start the SWE-bench evaluation docker (beware, huge size more than 30GB) provided by OpenDevin:

```bash
docker run -it \
-v /path/to/generation/outputs:/swe_bench_output \
ghcr.io/opendevin/eval-swe-bench:full-v1.0 /bin/bash
```

And run evaluation inside the docker to get the final evaluation stats:

```bash
export MINICONDA3=/swe_util/miniforge3
export OD_SWE_BENCH=/swe_util/OD-SWE-bench
export EVAL_DATA_DIR=/swe_util/eval_data

cd /swe_util && ./get_model_report.sh --output-file /swe_bench_output/model-name.json --model-name "model-name" --dataset swe-bench-test-lite
```


### Other Experiments

We also explored document chunking and reranking for better RACG.

#### Document Reranking
To reranking the top-100 retrieved documents and possibly get better top-k (k<<100) documents, we use an existing reranker model:
```
python rerank.py --results_path ${retrieval_results_file}
```
This script will prompt to ask the query field by providing a list of options available in the `results_path`.
For example, on the humaneval retrieval results, it will prompt
```
Choose query field from [dict_keys(['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point', 'docs'])]:
```
You can type in ("prompt" + Enter) and it will automatically finish the reranking.

For other datasets, use "text" for mbpp and livecodebench, "prompt" for ds-1000, "intent" for odex, "prompt" for repoeval, and "problem_statement" for swebench.


#### Document chunking
To take the first N-token chunk from the retrieved document, and process them as a new retrieval file:
```
python chunk.py --results_path ${retrieval_results_file} --max_num_tokens 500
```
We also support a heuristic-based chunking specifically for library documentation, to take only the beginning textual descriptions. To do this, run:
```
python chunk.py --results_path ${retrieval_results_file} --is_docs
```
