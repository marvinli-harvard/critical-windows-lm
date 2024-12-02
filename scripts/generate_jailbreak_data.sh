#!/bin/bash
#SBATCH --job-name=generate_jailbreak_data
#SBATCH -t 0-10:00                 # Runtime in D-HH:MM
#SBATCH --mem=200000               # Memory pool for all cores
#SBATCH -p gpu,gpu_requeue,seas_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --array=0-1
#SBATCH --mail-user=marvinli@college.harvard.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -o /n/holyscratch01/sitanc_lab/mfli/slurm_logs/logs/output/myoutput_%A_%a.out   # STDOUT with task-specific path
#SBATCH -e /n/holyscratch01/sitanc_lab/mfli/slurm_logs/logs/err/myerrors_%A_%a.err    # STDERR with task-specific path

# Run the appropriate command based on the task ID
task_id=${SLURM_ARRAY_TASK_ID}

if [ "$task_id" -eq 0 ]; then
    python experiments/jailbreak/run_jailbreak_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct  --dataset_type repeat_word --num_per_noise 100
elif [ "$task_id" -eq 1 ]; then
    python experiments/jailbreak/run_jailbreak_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset_type prefill_attack --num_per_noise 1
else
    echo "Invalid SLURM_ARRAY_TASK_ID: $task_id"
fi

