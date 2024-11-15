#!/bin/bash
#SBATCH -t 0-24:00                 # Runtime in D-HH:MM
#SBATCH --mem=100000               # Memory pool for all cores
#SBATCH -p gpu,gpu_requeue,seas_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --array=0-5                # Array index to handle each dataset/task
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=marvinli@college.harvard.edu
#SBATCH -o logs/output/myoutput_%A_%a.out   # STDOUT with task-specific path
#SBATCH -e logs/err/myerrors_%A_%a.err    # STDERR with task-specific path

# Array of model commands and dataset names for organization
task_id=${SLURM_ARRAY_TASK_ID}

# Run the appropriate command based on the task ID
if [ "$task_id" -eq 0 ]; then
    python run_evalnoisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --answer_type multiple_choice \
        --experiment_dir results/NoiseDenoise/NoiseDenoise_model=meta-llama-Llama-3.1-8B-Instruct_dataset=truthfulqa-truthful_qa_split=validation_nsamples=2000     
elif [ "$task_id" -eq 1 ]; then
   python run_evalnoisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --answer_type multiple_choice \
        --experiment_dir results/NoiseDenoise/NoiseDenoise_model=meta-llama-Llama-3.1-8B-Instruct_dataset=allenai-ai2_arc-ARC-Challenge_split=train_nsamples=2000
elif [ "$task_id" -eq 2 ]; then
    python run_evalnoisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --answer_type multiple_choice \
        --experiment_dir /n/holylabs/LABS/sitanc_lab/Users/mfli/critical-windows-lm/results/NoiseDenoise/NoiseDenoise_model=meta-llama-Llama-3.1-8B-Instruct_dataset=allenai-ai2_arc-ARC-Easy_split=train_nsamples=2000
elif [ "$task_id" -eq 3 ]; then
    python run_evalnoisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --answer_type multiple_choice \
        --experiment_dir /n/holylabs/LABS/sitanc_lab/Users/mfli/critical-windows-lm/results/NoiseDenoise/NoiseDenoise_model=meta-llama-Llama-3.1-8B-Instruct_dataset=cais-mmlu_split=auxiliary_train_nsamples=2000
elif [ "$task_id" -eq 4 ]; then
    python run_evalnoisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --answer_type multiple_choice \
        --experiment_dir /n/holylabs/LABS/sitanc_lab/Users/mfli/critical-windows-lm/results/NoiseDenoise/NoiseDenoise_model=meta-llama-Llama-3.1-8B-Instruct_dataset=lucasmccabe-logiqa_split=train_nsamples=2000
elif [ "$task_id" -eq 5 ]; then
    python run_evalnoisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --answer_type multiple_choice \
        --experiment_dir /n/holylabs/LABS/sitanc_lab/Users/mfli/critical-windows-lm/results/NoiseDenoise/NoiseDenoise_model=meta-llama-Llama-3.1-8B-Instruct_dataset=competition_math_split=train_nsamples=2000
else
    echo "Invalid SLURM_ARRAY_TASK_ID: $task_id"
fi



