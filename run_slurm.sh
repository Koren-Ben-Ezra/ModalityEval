#!/bin/bash
#SBATCH --job-name=ModalityEval
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --mem=5G                # Fixed: use 5G instead of 5000
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err

# Redirect stdout/stderr to specific files if args provided
if [[ $# -ge 2 ]]; then
  exec 1> outputs/${1}_${2}_Amit.out
  exec 2> outputs/${1}_${2}_Amit.err
fi

# Load CUDA drivers (must be before any CUDA call)
module purge
module load cuda/11.0

# Check GPU assignment
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Input args
export SELECTED_EVAL=$1
export SELECTED_TASK=$2

# Activate Conda environment
source /home/joberant/NLP_2425a/$(whoami)/anaconda3/etc/profile.d/conda.sh
conda activate ModalityEval

# Sanity check PyTorch GPU access
python -c "import torch; print('cuda available?', torch.cuda.is_available(), 'cuda version:', torch.version.cuda)"

# Hugging Face + scratch storage setup
export HUGGING_FACE_HUB_TOKEN="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"
export HF_HOME=/vol/scratch/$(whoami)/hf_cache; mkdir -p "$HF_HOME"
export TMPDIR=/vol/scratch/$(whoami)/tmp; mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUM_PARALLEL_JOBS=1

# Run main program
python -u main.py
