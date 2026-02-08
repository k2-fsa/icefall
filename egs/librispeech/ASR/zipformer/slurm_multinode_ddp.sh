#!/bin/bash -l
#
# Multi-node DDP training script for Zipformer using SLURM + torchrun
#
# This script demonstrates how to run distributed training across multiple
# nodes using SLURM as the job scheduler and PyTorch's torchrun for process
# management within each node.
#
# Usage:
#   sbatch run_multinode_ddp.sh
#
# Requirements:
#   - SLURM cluster with GPU nodes
#   - PyTorch with NCCL backend support
#   - Nodes must be able to communicate over TCP (for NCCL)
#
# Adjust SBATCH directives and training arguments below to match your setup.

#SBATCH -J zipformer-ddp
#SBATCH -o logs/zipformer_ddp_%N_%j.log
#SBATCH -p gpu                           # Partition name (adjust to your cluster)
#SBATCH --nodes=2                        # Number of nodes
#SBATCH --ntasks-per-node=1              # 1 torchrun launcher per node
#SBATCH --gpus-per-node=8                # GPUs per node
#SBATCH -c 24                            # CPU cores per task
#SBATCH --mem=0                          # Use all available memory

set -euo pipefail

# ============================================================================
# Environment setup
# ============================================================================

# Activate your conda environment (adjust path as needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate k2-icefall

# Set PYTHONPATH to include icefall
export PYTHONPATH=$PWD/../../..:${PYTHONPATH:-}

# ============================================================================
# Debugging options (optional, can be removed for production runs)
# ============================================================================

# Uncomment for verbose NCCL debugging
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

# Disable InfiniBand if your cluster uses Ethernet
# (comment out if your cluster has InfiniBand support)
export NCCL_IB_DISABLE=1

# ============================================================================
# Distributed training configuration
# ============================================================================

echo "Running on nodes: ${SLURM_JOB_NODELIST}"
HOSTS=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
MASTER_NODE="${HOSTS[0]}"
echo "Master node is: ${MASTER_NODE}"

# Get master node's IP address
MASTER_ADDR=$(srun -N1 -n1 -w "${MASTER_NODE}" bash -lc \
  "ip -o -4 addr show scope global | awk '{print \$4}' | cut -d/ -f1 | head -n1")

# Use a job-unique port to avoid collisions with other jobs
MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))

export MASTER_ADDR MASTER_PORT

# Calculate world size
GPUS_PER_NODE=8
WORLD_SIZE=$(( SLURM_NNODES * GPUS_PER_NODE ))

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "WORLD_SIZE=${WORLD_SIZE}"

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================================================
# Training configuration - MODIFY THESE FOR YOUR EXPERIMENT
# ============================================================================

EXP_DIR="zipformer/exp-multinode"
BPE_MODEL="data/lang_bpe_500/bpe.model"
NUM_EPOCHS=30
MAX_DURATION=1000

# For streaming model, set CAUSAL=1
CAUSAL=0
CHUNK_SIZE="16,32,64,-1"
LEFT_CONTEXT_FRAMES="64,128,256,-1"

# ============================================================================
# Launch training
# ============================================================================

# Launch exactly 1 torchrun process per node
# Each torchrun will spawn GPUS_PER_NODE worker processes
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 --kill-on-bad-exit=1 --export=ALL bash -lc '
  set -euo pipefail
  
  # Re-activate environment in the srun context
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate k2-icefall
  export PYTHONPATH='"$PWD"'/../../..:${PYTHONPATH:-}

  echo "Host=$(hostname) SLURM_PROCID=$SLURM_PROCID SLURM_NODEID=${SLURM_NODEID:-NA}"

  # Determine if this node should host the rendezvous server
  # Only the master node (SLURM_PROCID=0) hosts the TCPStore
  if [ "$SLURM_PROCID" -eq 0 ]; then
    RDZV_IS_HOST=1
  else
    RDZV_IS_HOST=0
    # Small delay to ensure master is ready
    sleep 5
  fi

  torchrun \
    --nnodes='"$SLURM_NNODES"' \
    --node_rank="$SLURM_PROCID" \
    --nproc_per_node='"$GPUS_PER_NODE"' \
    --rdzv_id='"$SLURM_JOB_ID"' \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR"':'"$MASTER_PORT"' \
    --rdzv_conf is_host="$RDZV_IS_HOST" \
    --max_restarts 0 \
    ./zipformer/train.py \
      --world-size '"$WORLD_SIZE"' \
      --num-epochs '"$NUM_EPOCHS"' \
      --use-fp16 1 \
      --exp-dir '"$EXP_DIR"' \
      --max-duration '"$MAX_DURATION"' \
      --causal '"$CAUSAL"' \
      --chunk-size '"$CHUNK_SIZE"' \
      --left-context-frames '"$LEFT_CONTEXT_FRAMES"' \
      --full-libri 1 \
      --bpe-model '"$BPE_MODEL"'
'

echo "Training complete!"
