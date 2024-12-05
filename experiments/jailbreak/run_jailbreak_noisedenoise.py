import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import argparse

import torch
from accelerate.utils import set_seed

from utils.utils import *
from utils.configuration import *
from utils.dataset_utils import *
from utils.generation_utils import *
from utils.grader_utils import *

from generation.GenerateEvalBase import *
from generation.GenerateEvalExamples import *

"""
Runs generations to produce critical windows on jailbreaks
python experiments/jailbreak/run_jailbreak_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
    --dataset_type repeat_word --num_per_noise 10

python experiments/jailbreak/run_jailbreak_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
    --dataset_type prefill_attack --num_per_noise 10 --num_samples 10
"""

def main():
    ####################################################################################################
    # SETUP
    ####################################################################################################
    parser = argparse.ArgumentParser()
    ## Model arguments
    parser.add_argument('--model_id', action="store", type=str, required=True, help="Name of model")

    ## Experiment config details
    parser.add_argument('--experiment_name', action="store", type=str, required=False, help='Optional experiment name.')
    parser.add_argument('--max_gen_length', action="store", type=int, required=False, default=JAILBREAK_MAX_LEN,help='Maximum generation length')
    parser.add_argument('--num_per_noise', action="store", type=int, required=False, default=1,help='Number of samples per `noise level`.')
    
    ## Dataset arguments
    parser.add_argument('--dataset_type', action="store", type=str, required=True, 
                        choices=[e.value for e in DatasetType], help='Type of fixed noise denoise dataset')
    parser.add_argument('--num_samples', action="store", type=int, required=False, help='Number of samples.')

    
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=DEFAULT_SEED, help='Seed')
    args = parser.parse_args()
    
    start_time = time.time()

    ## Construct config file and save 
    if args.experiment_name is None:
        args.experiment_name = f"JailbreakNoiseDenoise_model={args.model_id.replace('/','-')}_dataset_type={args.dataset_type}_num_samples={args.num_samples}_num_per_noise={args.num_per_noise}"        
    args.experiment_dir  = f"results/JailbreakNoiseDenoise/{args.experiment_name}/"
    args.date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    os.makedirs(os.path.dirname(args.experiment_dir), exist_ok=True)
    
    # Save to JSON
    with open(f"{args.experiment_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    ############################################################################################################################################################################################################################################
    ## Load model and dataset
    ############################################################################################################################################################################################################################################
    set_seed(args.seed)

    ## Construct dataset and load model
    model = load_all(model_id=args.model_id,max_gen_length=args.max_gen_length,num_per_noise=args.num_per_noise,seed=args.seed)
    if args.dataset_type == DatasetType.REPEAT_WORD.value:
        dataset = create_repetition_dataset(tokenizer=model.tokenizer)
        generator = GenerateEvalRepeat(genwrapper = model, tokenizer=model.tokenizer)
    elif args.dataset_type == DatasetType.PREFILL_ATTACK.value:
        dataset = create_prefill_jailbreak_dataset(tokenizer=model.tokenizer)
        generator = GenerateEvalJailbreak(genwrapper = model, tokenizer=model.tokenizer)
    if args.num_samples:
        dataset = dataset.shuffle().select(range(min(args.num_samples, len(dataset))))
    ############################################################################################################################################################################################################################################
    ## Run evaluation and grade each answer
    ############################################################################################################################################################################################################################################
    print("Generating responses")
    model_answers = generator.generate_responses(dataset, model.sampling_repeat)
    
    ## Delete model
    del model.model
    torch.cuda.empty_cache()

    print("Grading responses")
    graded_answers = generator.grade(model_answers )
    
    # Save graded answers to a .pt file
    torch.save(graded_answers, f"{args.experiment_dir}/responses_graded.pt")

    print(f"Experiment completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
