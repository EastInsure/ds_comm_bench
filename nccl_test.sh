#!/bin/bash
#set -x

export OMP_NUM_THREADS=16

MASTER_ADDR=$1
MASTER_PORT=$2
NNODES=$3
NODE_RANK=$4
custom_cmd=$5

GPUS_PER_NODE=8

export CMD=${custom_cmd:-"run_all.py"}

#     --deepspeed-activation-checkpointing \

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    "

export LAUNCHER="python -u -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK \
    "

export NODE_RANK
mkdir -p $(dirname $0)/logs
logfile=$(dirname $0)/logs/$NNODES-${GPUS_PER_NODE}-${HOSTNAME}.log

echo "SCRIPT_CMD:$CMD"
echo "MASTER_ADDR:$MASTER_ADDR MASTER_PORT:$MASTER_PORT NNODES:$NNODES NODE_RANK:$NODE_RANK"
echo "LOGFILE:$logfile"

logfile=$(dirname $0)/logs/$NNODES-${GPUS_PER_NODE}-${HOSTNAME}.log
bash -c '$LAUNCHER  $CMD' > >(tee $logfile) 2>&1


