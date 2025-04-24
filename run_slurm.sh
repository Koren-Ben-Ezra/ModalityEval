#!/bin/bash
#SBATCH --job-name=ModalityEval
#SBATCH --partition=studentkillable
#SBATCH --mem=5G
#SBATCH --cpus-per-task=2
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err

# Optional: custom output files if args passed
if [[ $# -ge 2 ]]; then
  exec 1> outputs/${1}_${2}_Amit.out
  exec 2> outputs/${1}_${2}_Amit.err
fi

# Parse input arguments
export SELECTED_EVAL=$1
export SELECTED_TASK=$2

# Print basic info for logs
echo "Running on node: $(hostname)"
echo "Date: $(date)"
echo "Args: SELECTED_EVAL=$SELECTED_EVAL, SELECTED_TASK=$SELECTED_TASK"
echo "CPU Count: $SLURM_CPUS_ON_NODE"

# Hugging Face + scratch setup (if needed)
export HUGGING_FACE_HUB_TOKEN="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"
export HF_HOME=/vol/scratch/$(whoami)/hf_cache; mkdir -p "$HF_HOME"
export TMPDIR=/vol/scratch/$(whoami)/tmp; mkdir -p "$TMPDIR"

# Run your CPU-based script
python -u main.py
