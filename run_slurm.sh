#!/bin/bash
#SBATCH --job-name=MultimodalEval
#SBATCH --output=outputs/output.out    # Job name and ID in output filename for easier tracking
#SBATCH --error=outputs/error.err     # Ditto for error files
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --gres=gpu:5                  # Request a total of 5 GPUs
#SBATCH --ntasks=5                    # Launch 5 tasks (one per GPU)
#SBATCH --mem=50000                 # Request 50GB of CPU memory
#SBATCH --constraint=geforce_rtx_3090

# Display allocated GPU information
echo "Allocated GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# --- Environment Setup ---

# (Optional) If your system uses a module system to load CUDA, you may uncomment the next line:
# module load cuda/11.0

# If the module system isn't used, you can set CUDA paths manually:
# export PATH=/usr/local/cuda-11.0/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH

# Initialize Conda and activate the environment
source /home/joberant/NLP_2425a/$(whoami)/anaconda3/etc/profile.d/conda.sh
conda activate ModalityEval

# Set Hugging Face hub token (if using HF models)
export HUGGING_FACE_HUB_TOKEN="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"

# Configure Hugging Face cache and temporary directory in scratch space to avoid home quota issues
export HF_HOME=/vol/scratch/$(whoami)/hf_cache
mkdir -p "$HF_HOME"
export TMPDIR=/vol/scratch/$(whoami)/tmp
mkdir -p "$TMPDIR"

# Optional: Configure PyTorch CUDA allocation for managing memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Run Benchmark Tasks in Parallel ---

# Launch 5 parallel tasks using srun.
# Each task is assumed to run on one GPU. Your main.py should be written to operate independently per GPU.
srun --ntasks=1 python -u main.py &
srun --ntasks=1 python -u main.py &
srun --ntasks=1 python -u main.py &
srun --ntasks=1 python -u main.py &
srun --ntasks=1 python -u main.py &

# Wait for all background tasks to complete
wait
