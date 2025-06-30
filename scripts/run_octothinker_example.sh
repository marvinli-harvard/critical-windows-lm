#!/bin/bash
#SBATCH --job-name=octothinker_qa_test
#SBATCH --output=err_out/octothinker_qa_test.out
#SBATCH --error=err_out/octothinker_qa_test.err
#SBATCH --partition gpu,seas_gpu
#SBATCH --gres gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --account sitanc_lab
#SBATCH --mail-type ALL
#SBATCH --mail-user marvinli@college.harvard.edu
#SBATCH --comment "Test run with OctoThinker model on EleutherAI/hendrycks_math"

export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p err_out

# Activate conda environment
conda activate coverage

# Run QA noisedenoise experiment with OctoThinker-3B-Long-Base
echo "Running QA noisedenoise experiment with OctoThinker-3B-Long-Base..."
python experiments/chain_of_thought/run_qa_noisedenoise.py \
    --model_id OctoThinker/OctoThinker-3B-Long-Base \
    --dataset EleutherAI/hendrycks_math \
    --split test \
    --task math \
    --num_samples 100 \
    --num_per_noise 4 \
    --answer_type math \
    --percent_prompt 0.0 \
    --max_gen_length 10000

echo "Completed experiment with OctoThinker-3B-Long-Base" 