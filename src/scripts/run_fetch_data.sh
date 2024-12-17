#!/bin/bash

TOTAL_TOKEN_NUM=# TODO: Your total token number, e.g., 40
CN_RATIO=# TODO: Your Chinese ratio, e.g., 0.1
EN_RATIO=# TODO: Your English ratio, e.g., 0.7
SYN_RATIO=# TODO: Your synthetic data ratio, e.g., 0.2
ROOT_DIR=# TODO: Your data root directory, e.g., /data
TOKENIZER_PATH=# TODO: Your tokenizer path, e.g., meta-llama/Meta-Llama-3-8B

python fetch_data.py \
    --total_token_num ${TOTAL_TOKEN_NUM} \
    --cn_ratio ${CN_RATIO} \
    --en_ratio ${EN_RATIO} \
    --syn_ratio ${SYN_RATIO} \
    --root_dir ${ROOT_DIR} \
    --tokenizer_path ${TOKENIZER_PATH} \
