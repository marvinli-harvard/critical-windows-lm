#!/bin/bash
#SBATCH --job-name=generate_jailbreak_likelihood_ratio
#SBATCH -t 0-02:00                 # Runtime in D-HH:MM
#SBATCH --mem=200000               # Memory pool for all cores
#SBATCH -p gpu,gpu_requeue,seas_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mail-user=marvinli@college.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-1                # Job array with 2 tasks (adjust range as needed)
#SBATCH -o /n/netscratch/sitanc_lab/Lab/mfli/slurm_logs/logs/out/%A_%a.out   # STDOUT with task-specific path
#SBATCH -e /n/netscratch/sitanc_lab/Lab/mfli/slurm_logs/logs/err/%A_%a.err   # STDERR with task-specific path

# Task definitions based on the job array ID
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python experiments/jailbreak/run_likelihood_ratio_jailbreak.py --aligned_model meta-llama/Llama-3.1-8B-Instruct \
        --unaligned_model grimjim/Llama-3.1-8B-Instruct-abliterated_via_adapter
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    python experiments/jailbreak/run_likelihood_ratio_jailbreak.py --aligned_model meta-llama/Llama-3.1-8B-Instruct \
        --unaligned_model meta-llama/Llama-3.1-8B
else
    echo "Invalid task ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi


