#!/bin/bash

NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4
JOB_NAME=$5
SCRIPT_PATH=$6
MODEL_PATH=$7
DATA_PATH=$8
SAVE_DIR=$9

export OMP_NUM_THREADS=24
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0

MODEL_MAX_LENGTH=# TODO: Set model max length, e.g., 8192
PER_DEVICE_TRAIN_BATCH_SIZE=# TODO: Set per device train batch size, e.g., 2
GRADIENT_ACCUMULATION_STEPS=# TODO: Set gradient accumulation steps, e.g., 1
SAVE_STEPS=# TODO: Set model save steps, e.g., 2000
SAVE_TOTAL_LIMIT=# TODO: Set model save total limit, e.g., 10
LEARNING_RATE=# TODO: Set learning rate, e.g., 1e-5
WARMUP_RATIO=# TODO: Set warmup ratio, e.g., 0.0
WEIGHT_DECAY=# TODO: Set weight decay, e.g., 0.0
DEEPSPEED_CONFIG_PATH=# TODO: Set deepspeed config path, e.g., ./config/ds2_config.json
GRADIENT_CHECKPOINTING=# TODO: Whether to use gradient checkpointing, e.g., True
# Set `--flash_attention` to use FlashAttention 2.
# Set `--use_wsd` to use the WSD optimizer.
# Set `--no_shuffle` to disable shuffling.
# Set `--load_text_dataset` to load raw text data and perform preprocessing (tokenization and grouping) before training. After preprocessing, the script will save the processed dataset to disk and exit. If False, it assumes that a preprocessed dataset is provided and loads it directly from disk.
# Set `single_dataset` to load a single dataset from the specified `data_path`. If False, it will load and concatenate multiple datasets found in the `data_path` directory.
# Set `--resume_from_checkpoint <checkpoint_path>` to resume training from a checkpoint.


torchrun --nproc_per_node=8 \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    ${SCRIPT_PATH} \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --bf16 True \
    --output_dir ${SAVE_DIR}/${JOB_NAME} \
    --num_train_epochs 1 \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --logging_steps 2 \
    --deepspeed ${DEEPSPEED_CONFIG_PATH} \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --report_to none \
    --tf32 True \
    --lr_scheduler_type "linear" \
    --flash_attention \
    --use_wsd \
&> ./log/${JOB_NAME}_part${NODE_RANK}.log
