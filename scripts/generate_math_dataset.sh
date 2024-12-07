#!/bin/bash
#SBATCH --job-name=generate_math_dataset
#SBATCH -t 0-30:00                 # Runtime in D-HH:MM
#SBATCH --mem=200000               # Memory pool for all cores
#SBATCH -p gpu,gpu_requeue,seas_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mail-user=marvinli@college.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o /n/netscratch/sitanc_lab/Lab/mfli/slurm_logs/logs/out/out_%A_%a.out   # STDOUT with task-specific path
#SBATCH -e /n/netscratch/sitanc_lab/Lab/mfli/slurm_logs/logs/err/err_%A_%a.err    # STDERR with task-specific path



python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
        --dataset competition_math --split train --task math --num_samples 7500 --num_per_noise 10 --answer_type math

python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
        --dataset competition_math --split test --task math --num_samples 5000 --num_per_noise 10 --answer_type math


