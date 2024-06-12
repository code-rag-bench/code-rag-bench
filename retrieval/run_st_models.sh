MODEL_NAME=$1
MODEL_NAME_TAG=$2
BATCH_SIZE=$3

mkdir ${MODEL_NAME_TAG}_retrieval_results

python3 eval_beir_sbert.py \
    --dataset repoeval_retrieval_data/api \
    --model ${MODEL_NAME} \
    --output_file ${MODEL_NAME_TAG}_retrieval_results/repoeval_api_score.json \
    --results_file ${MODEL_NAME_TAG}_retrieval_results/repoeval_api_retrieval_results.json \
    --dataset_path  output/repoeval/datasets/api_level_completion_2k_context_codegen.test.jsonl
    --batch_size ${BATCH_SIZE}

python3 eval_beir_sbert.py \
    --dataset repoeval_retrieval_data/function \
    --model ${MODEL_NAME} \
    --output_file ${MODEL_NAME_TAG}_retrieval_results/repoeval_function_score.json \
    --results_file ${MODEL_NAME_TAG}_retrieval_results/repoeval_function_retrieval_results.json \
    --dataset_path  output/repoeval/datasets/function_level_completion_2k_context_codex.test.jsonl
    --batch_size ${BATCH_SIZE}

python3 eval_beir_sbert.py \
    --dataset livecodebench \
    --model ${MODEL_NAME} \
    --output_file ${MODEL_NAME_TAG}_retrieval_results/livecodebench_score.json \
    --results_file ${MODEL_NAME_TAG}_retrieval_results/livecodebench_retrieval_results.json \
    --batch_size ${BATCH_SIZE}

python3 eval_beir_sbert.py \
    --dataset swe-bench-lite \
    --model ${MODEL_NAME} \
    --output_file ${MODEL_NAME_TAG}_retrieval_results/swe-bench-lite_score.json \
    --results_file ${MODEL_NAME_TAG}_retrieval_results/swe-bench-lite_retrieval_results.json \
    --batch_size ${BATCH_SIZE}

python3 eval_beir_sbert.py \
    --dataset mbpp \
    --model ${MODEL_NAME} \
    --output_file ${MODEL_NAME_TAG}_retrieval_results/mbpp_score.json \
    --results_file ${MODEL_NAME_TAG}_retrieval_results/mbpp_retrieval_results.json \
    --batch_size ${BATCH_SIZE}

python3 eval_beir_sbert.py \
    --dataset humaneval \
    --model ${MODEL_NAME} \
    --output_file ${MODEL_NAME_TAG}_retrieval_results/humaneval_score.json \
    --results_file ${MODEL_NAME_TAG}_retrieval_results/humaneval_retrieval_results.json \
    --batch_size ${BATCH_SIZE}

python3 eval_beir_sbert.py \
    --dataset ds1000_all_completion \
    --model ${MODEL_NAME} \
    --output_file ${MODEL_NAME_TAG}_retrieval_results/ds1000_all_completion_score.json \
    --results_file ${MODEL_NAME_TAG}_retrieval_results/ds1000_all_completion_retrieval_results.json \
    --batch_size ${BATCH_SIZE}

python3 eval_beir_sbert.py \
    --dataset odex_en \
    --model ${MODEL_NAME} \
    --output_file ${MODEL_NAME_TAG}_retrieval_results/odex_en_all_completion_score.json \
    --results_file ${MODEL_NAME_TAG}_retrieval_results/odex_en_all_completion_retrieval_results.json \
    --batch_size ${BATCH_SIZE}
