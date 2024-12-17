#!/bin/bash

TIMESTAMP_LST=# TODO: Your timestamp list provided by fetch_data.py
TOKENIZER_PATH=# TODO: Your tokenizer path, e.g., meta-llama/Meta-Llama-3-8B
MODEL_MAX_LENGTH=# TODO: Your model max length, e.g., 8192
NUM_WORKERS=# TODO: Your number of workers, e.g., 32
MIN_TEXT_LENGTH=# TODO: Your minimum text length, e.g., 10
ROOT_DIR=# TODO: Your data root directory, e.g., /data

python save_hf_dataset.py \
    --timestamp_lst ${TIMESTAMP_LST} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --num_workers ${NUM_WORKERS} \
    --min_text_length ${MIN_TEXT_LENGTH} \
    --root_dir ${ROOT_DIR} \
    --show_case # Show the first case
