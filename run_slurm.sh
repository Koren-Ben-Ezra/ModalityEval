#!/bin/bash
#SBATCH --job-name=ModalityEval
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --mem=5000
ß
exec 1> outputs/${1}_${2}_Amit.out
exec 2> outputs/${1}_${2}_Amit.err

# Load the cluster’s driver/toolkit before any nvidia-smi or torch.cuda calls
module purge
module load cuda/11.0

# Now you can safely query the GPU
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv

export SELECTED_EVAL=$1
export SELECTED_TASK=$2

# Initialize Conda and activate your environment
source /home/joberant/NLP_2425a/$(whoami)/anaconda3/etc/profile.d/conda.sh
conda activate ModalityEval

# Sanity check PyTorch
python -c "import torch; print('cuda available?', torch.cuda.is_available(), 'cuda version:', torch.version.cuda)"

export HUGGING_FACE_HUB_TOKEN="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"

# HuggingFace cache & tmp
export HF_HOME=/vol/scratch/$(whoami)/hf_cache; mkdir -p "$HF_HOME"
export TMPDIR=/vol/scratch/$(whoami)/tmp; mkdir -p "$TMPDIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUM_PARALLEL_JOBS=1

# Finally run
python -u main.py
