#!/bin/bash

PWD=`pwd`

### Execute your job
SCRIPT=run_train_multi.sh
SCRIPT_PATH=${PWD}/train.py
MODEL_NAME=# TODO: model name
MODEL_PATH=# TODO: model path
DATA_PREFIX=# TODO: data prefix
DATA_PATH=# TODO: data path
JOB_NAME=${MODEL_NAME}_CPT_${DATA_PREFIX}
SAVE_DIR=# TODO: save dir, e.g., ${PWD}/model
nodelist=(\
    # "job-master-0" \
    # "job-worker-0" \
)
MASTER_PORT=# TODO: master port

NNODES="${#nodelist[@]}"
LOCAL_HOST=`hostname`
echo "'"$LOCAL_HOST"'" $NNODES $MASTER_PORT

for ((i=0;i<${NNODES};i=i+1))
do
    echo "${nodelist[i]} => " "cd ${PWD} && bash ${SCRIPT} ${NNODES} $i ${nodelist[0]} ${MASTER_PORT} ${JOB_NAME} ${SCRIPT_PATH} ${MODEL_PATH} ${DATA_PATH} ${SAVE_DIR}" "&> ${PWD}/log/${JOB_NAME}_part${i}.log &"
    ssh -o ServerAliveInterval=60 "${nodelist[i]}" "cd ${PWD} && bash ${SCRIPT} ${NNODES} $i ${nodelist[0]} ${MASTER_PORT} ${JOB_NAME} ${SCRIPT_PATH} ${MODEL_PATH} ${DATA_PATH} ${SAVE_DIR}" &> ${PWD}/log/${JOB_NAME}_part${i}.log &
done

wait

echo finished!
