#!/usr/bin/env bash
#
# Launch zipformer training with torchrun (single- or multi-node).
# train.py reads RANK / LOCAL_RANK / WORLD_SIZE from the env, so
# `--world-size` is ignored here.
#
# Single 8-GPU node:  bash zipformer/run_train_cluster.sh
# Multi-node: the scheduler is expected to export WORLD_SIZE (#nodes),
# RANK (node rank), MASTER_ADDR and MASTER_PORT.

set -euo pipefail

export PYTHONPATH=`pwd`/../../../
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}   # GPUs per node
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_MIN_NCHANNELS=4
export NCCL_NET_PLUGIN=none
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  ./zipformer/train.py \
  --num-epochs 100 \
  --start-epoch 1 \
  --use-bf16 1 \
  --exp-dir zipformer/exp \
  --max-duration 5000
