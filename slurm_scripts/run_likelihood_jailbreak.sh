#!/bin/bash

#SBATCH --job-name=likelihood_jailbreak
#SBATCH -t 0-01:00                 # Runtime in D-HH:MM
#SBATCH --mem=100000               # Memory pool for all cores
#SBATCH -p gpu,gpu_requeue,seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-14%5                # Array index to handle each dataset/task
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=marvinli@college.harvard.edu
#SBATCH -o /n/holyscratch01/sitanc_lab/mfli/slurm_logs/logs/output/myoutput_%A_%a.out   # STDOUT with task-specific path
#SBATCH -e /n/holyscratch01/sitanc_lab/mfli/slurm_logs/logs/err/myerrors_%A_%a.err    # STDERR with task-specific path

splits=('benign_instructions_train' 'circuit_breakers_train' 'harmful_autodan' 'harmful_gcg' 'harmful_human_mt' 'harmful_instructions_test' 'harmful_instructions_train' 'harmful_misc' 'harmful_msj' 'harmful_pair' 'harmful_prefill' 'mt_bench' 'or_bench_train' 'wildchat' 'xstest')

split=${splits[$SLURM_ARRAY_TASK_ID]}
python run_likelihood_ratio_alignment.py --aligned_model meta-llama/Llama-3.1-8B-Instruct \
    --unaligned_model grimjim/Llama-3.1-8B-Instruct-abliterated_via_adapter   \
    --dataset Mechanistic-Anomaly-Detection/llama3-jailbreaks --split $split --bs 16
