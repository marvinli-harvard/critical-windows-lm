#!/bin/bash
#SBATCH --job-name=qa_noisedenoise_hendrycks_math
#SBATCH --output=err_out/qa_noisedenoise_hendrycks_math/qa_noisedenoise_hendrycks_math%A_%a.out
#SBATCH --error=err_out/qa_noisedenoise_hendrycks_math/qa_noisedenoise_hendrycks_math%A_%a.err
#SBATCH --partition gpu,seas_gpu
#SBATCH --gres gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --account sitanc_lab
#SBATCH --mail-type ALL
#SBATCH --mail-user marvinli@college.harvard.edu
#SBATCH --comment "QA noisedenoise experiments on EleutherAI/hendrycks_math with multiple models"

export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p err_out/qa_noisedenoise_hendrycks_math

# Define models (note: meta-llama-Llama becomes meta-llama/Llama)
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "OctoThinker/OctoThinker-3B-Long-Base"
    "OctoThinker/OctoThinker-3B-Long-Zero"
)

# Get model from array index
MODEL_ID=${MODELS[$SLURM_ARRAY_TASK_ID]}

# Activate conda environment
conda activate coverage

# Run QA noisedenoise experiment
echo "Running QA noisedenoise experiment with model: ${MODEL_ID}..."
python experiments/chain_of_thought/run_qa_noisedenoise.py \
    --model_id ${MODEL_ID} \
    --dataset EleutherAI/hendrycks_math \
    --split test \
    --task math \
    --num_samples 1000 \
    --num_per_noise 8 \
    --answer_type math \
    --percent_prompt 0.0 \
    --max_gen_length 10000 

echo "Completed experiment for model: ${MODEL_ID}" 
