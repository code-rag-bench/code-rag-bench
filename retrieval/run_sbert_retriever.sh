MODEL_NAME=$1
MODEL_TAG=$2
BATCH_SIZE=$3

for dataset_name in "humaneval" "mbpp" "apps" "ds1000_all_completion" "odex_en" "odex_es" "odex_ja" "odex_ru" "docprompting_conala" "code_search_net_ruby" "code_search_net_go" "code_search_net_java" "code_search_net_python" "code_search_net_javascript"
do
python eval_beir_sbert.py --dataset $dataset_name --model $MODEL_NAME --batch_size $BATCH_SIZE --output_file results/${MODEL_TAG}_${dataset_name}_output.json --results_file results/${MODEL_TAG}_${dataset_name}_results.json
done
