#!/bin/bash
#SBATCH --job-name=generate_synthetic_data
#SBATCH -t 0-24:00                 # Runtime in D-HH:MM
#SBATCH --mem=200000               # Memory pool for all cores
#SBATCH -p gpu,gpu_requeue,seas_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-6
#SBATCH --mail-user=marvinli@college.harvard.edu
#SBATCH -o /n/netscratch/sitanc_lab/Lab/mfli/slurm_logs/logs/out/out_%A_%a.out   # STDOUT with task-specific path
#SBATCH -e /n/netscratch/sitanc_lab/Lab/mfli/slurm_logs/logs/err/err_%A_%a.err    # STDERR with task-specific path

python experiments/synthetic_experiments/run_synthetic_experiments.py --model_id meta-llama/Llama-3.1-8B-Instruct --num_samples 10000

