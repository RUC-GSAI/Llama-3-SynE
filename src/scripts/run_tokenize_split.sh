#!/bin/bash

TOKENIZER_PATH=# TODO: Your tokenizer path, e.g., meta-llama/Meta-Llama-3-8B
DATA_PATH=# TODO: Your data path
MODEL_NAME=# TODO: Your model name

python tokenize_text.py \
    --tokenizer_path ${TOKENIZER_PATH} \
    --data_path ${DATA_PATH} \
    --model_name ${MODEL_NAME} \
    --num_file 500000 \
    --text_key text \
    --num_worker 64 \
    --skip_exist True

FATHER_DATASETS=# TODO: Your father datasets

python split_data.py \
    --father_datasets ${FATHER_DATASETS}
