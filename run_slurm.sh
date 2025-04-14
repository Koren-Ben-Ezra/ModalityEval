#!/bin/bash
#SBATCH --job-name=MultimodalEval
#SBATCH --output=outputs/output.out
#SBATCH --error=outputs/error.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --gres=gpu:5
#SBATCH --ntasks=5
#SBATCH --mem=50000         # 50,000 MB (50GB) of CPU memory
#SBATCH --constraint=geforce_rtx_3090

# Print GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv

# Manually set CUDA paths (if module system isn't used)
export PATH=/usr/local/cuda-11.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH

# Initialize Conda and activate your environment
source /home/joberant/NLP_2425a/$(whoami)/anaconda3/etc/profile.d/conda.sh
conda activate ModalityEval
export HUGGING_FACE_HUB_TOKEN="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"
# Set Hugging Face cache and TMPDIR to scratch space to avoid home quota issues
export HF_HOME=/vol/scratch/$(whoami)/hf_cache
mkdir -p "$HF_HOME"

export TMPDIR=/vol/scratch/$(whoami)/tmp
mkdir -p "$TMPDIR"

# Optional: Configure PyTorch CUDA allocation to help avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# Run your main Python script in unbuffered mode
python -u main.py

tail -f outputs/output.out