MODEL_NAME=$1
MODEL_ID=$2
for lang in python ruby java php go;
do
python eval_beir_sbert.py \
    --model ${MODEL_NAME} \
    --dataset code_search_net_${lang} \
    --output_file result_${MODEL_ID}_${lang}.json
done
