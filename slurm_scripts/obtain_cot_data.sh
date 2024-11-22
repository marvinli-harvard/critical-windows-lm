#!/bin/bash
#SBATCH --job-name=obtain_cot_data
#SBATCH -t 0-10:00                 # Runtime in D-HH:MM
#SBATCH --mem=100000               # Memory pool for all cores
#SBATCH -p gpu,gpu_requeue,seas_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-6      
#SBATCH --mail-user=marvinli@college.harvard.edu
#SBATCH -o /n/holyscratch01/sitanc_lab/mfli/slurm_logs/logs/output/myoutput_%A_%a.out   # STDOUT with task-specific path
#SBATCH -e /n/holyscratch01/sitanc_lab/mfli/slurm_logs/logs/err/myerrors_%A_%a.err    # STDERR with task-specific path

# Run the appropriate command based on the task ID
bs=64
task_id=${SLURM_ARRAY_TASK_ID}

# Run the appropriate command based on the task ID
if [ "$task_id" -eq 0 ]; then
    python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset competition_math --split train --bs $bs --task math --num_samples 10 --num_per_noise 100
elif [ "$task_id" -eq 1 ]; then
    python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset lucasmccabe/logiqa --split train --bs $bs --task logic --num_samples 10 --num_per_noise 100
elif [ "$task_id" -eq 2 ]; then
    python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset truthfulqa/truthful_qa --split validation --bs $bs  --task "true or false" --num_samples 10 --num_per_noise 100
elif [ "$task_id" -eq 3 ]; then
    python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset cais/mmlu --split auxiliary_train --bs $bs --task "multiple choice" --num_samples 10 --num_per_noise 100
elif [ "$task_id" -eq 4 ]; then
    python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset allenai/ai2_arc@ARC-Challenge --split train --bs $bs  --task science --num_samples 10  --num_per_noise 100
elif [ "$task_id" -eq 5 ]; then
    python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset allenai/ai2_arc@ARC-Easy --split train --bs $bs --task science --num_samples 10  --num_per_noise 100
elif [ "$task_id" -eq 6 ]; then
    python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset deepmind/aqua_rat --split train --bs $bs  --task math --num_samples 10  --num_per_noise 100
else
    echo "Invalid SLURM_ARRAY_TASK_ID: $task_id"
fi


