#!/bin/bash
set -euo pipefail

# Usage: ./run.sh path/to/opt.yaml experiment_name [gpu_id]

opt=$1
name=$2
gpu_id=${3:-}

# Set CUDA_VISIBLE_DEVICES if gpu_id is provided
if [ -n "$gpu_id" ]; then
    export CUDA_VISIBLE_DEVICES=$gpu_id
    echo "Using GPU: $gpu_id"
else
    echo "Using default single-GPU configuration"
fi

# Run training (single-GPU by default)
python train.py --opt "$opt" --name "$name"

# Run evaluation
python eval.py --name "$name" --ckpt last