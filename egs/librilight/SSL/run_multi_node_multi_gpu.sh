#!/usr/bin/env bash
#
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Yifan Yang)
#
# This script is the entry point to start model training
# with multi-node multi-GPU.
#
# Read the usage instructions below for how to run this script.

set -e

# DDP related parameters
master_addr=
node_rank=
num_nodes=4
master_port=12354

. shared/parse_options.sh

function usage() {
  echo "Usage: "
  echo ""
  echo "    $0 \\"
  echo "      --master-addr <IP of master> \\"
  echo "      --master-port <Port of master> \\"
  echo "      --node-rank <rank of this node> \\"
  echo "      --num-nodes <Number of node>"
  echo ""
  echo " --master-addr   The ip address of the master node."
  echo " --master-port   The port of the master node."
  echo " --node-rank     Rank of this node."
  echo " --num-nodes     Number of nodes in DDP training."
  echo ""
  echo "Usage example:"
  echo "Suppose you want to use DDP with two machines:"
  echo "  (1) Machine 1 has 4 GPUs. You want to use"
  echo "      GPU 0, 1, and 3 for training"
  echo "      IP of machine 1 is: 10.177.41.71"
  echo "  (2) Machine 2 has 4 GPUs. You want to use"
  echo "      GPU 0, 2, and 3 for training"
  echo "      IP of machine 2 is: 10.177.41.72"
  echo "You want to select machine 1 as the master node and"
  echo "assume that the port 1234 is free on machine 1."
  echo ""
  echo "On machine 1, you run:"
  echo ""
  echo "  export CUDA_VISIBLE_DEVICES=\"0,1,3\""
  echo "  ./run_multi_node_multi_gpu.sh --master-addr 10.177.41.71 --master-port 1234 --node-rank 0 --num-nodes 2"
  echo ""
  echo "On machine 2, you run:"
  echo ""
  echo "  export CUDA_VISIBLE_DEVICES=\"0,2,3\""
  echo "  ./run_multi_node_multi_gpu.sh --master-addr 10.177.41.71 --master-port 1234 --node-rank 1 --num-nodes 2"
  echo ""
  echo "Note 1:"
  echo "  You use CUDA_VISIBLE_DEVICES to decide which GPUs are used for training."
  echo ""
  echo "Note 2:"
  echo "  If you use torch < 1.9.0, then every node has to use the same number of GPUs for training."
  echo "  If you use torch >= 1.9.0, different nodes can have a different number of GPUs for training."
  exit 1
}

default='\033[0m'
bold='\033[1m'
red='\033[31m'

function error() {
  printf "${bold}${red}[ERROR]${default} $1\n"
}

[ ! -z $CUDA_VISIBLE_DEVICES ] || ( echo; error "Please set CUDA_VISIBLE_DEVICES"; echo; usage )
[ ! -z $master_addr ] || ( echo; error "Please set --master-addr"; echo; usage )
[ ! -z $master_port ] || ( echo; error "Please set --master-port"; echo; usage )
[ ! -z $node_rank ] || ( echo; error "Please set --node-rank"; echo; usage )
[ ! -z $num_nodes ] || ( echo; error "Please set --num-nodes"; echo; usage )

# Number of GPUs this node has
num_gpus=$(python3 -c "s=\"$CUDA_VISIBLE_DEVICES\"; print(len(s.split(',')))")

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "num_gpus: $num_gpus"
echo "master_addr: $master_addr"

export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port

set -x

torchrun \
  --nproc_per_node $num_gpus \
  --nnodes $num_nodes \
  --node_rank $node_rank \
  --master_addr $master_addr \
  --master_port $master_port \
  zipformer/pretrain.py \
    --use-multi-node 1 \
    --master-port $master_port \
    --num-epochs 20 \
    --start-epoch 1 \
    --use-fp16 1 \
    --exp-dir zipformer/exp_pretrain \
    --max-duration 350 \
    --quadratic-duration 1024 \
    --accum-grad 1 \
    --do-normalize 1 \
    --mask-prob 0.8 \
    --dropout-input 0.0 \
    --dropout-features 0.0 \
    --feature-grad-mult 1.0 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 768,1536,2048,3072,2048,1536 \
    --encoder-dim 256,512,768,1024,768,512 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
    --base-lr 0.045
