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
Runs generations to produce critical windows on synthetic data
python experiments/synthetic_experiments/run_synthetic_experiments.py --model_id meta-llama/Llama-3.1-8B-Instruct --num_samples 10000
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
    parser.add_argument('--max_gen_length', action="store", type=int, required=False, default=MAX_GEN_LEN_COT,help='Maximum generation length')
    parser.add_argument('--story_str', action="store", type=str, required=False, default=SYNTHETIC_STORY_PROMPT,help='Synthetic story')
    parser.add_argument('--num_samples', action="store", type=int, required=False, default=10000, help='Number of samples.')

    
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=DEFAULT_SEED, help='Seed')
    args = parser.parse_args()
    
    start_time = time.time()

    ## Construct config file and save 
    if args.experiment_name is None:
        args.experiment_name = f"SyntheticNoiseDenoise_model={args.model_id.replace('/','-')}_num_samples={args.num_samples}"        
    args.experiment_dir  = f"results/SyntheticNoiseDenoise/{args.experiment_name}/"
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
    genwrapper = load_all(model_id=args.model_id,max_gen_length=args.max_gen_length,num_per_noise=args.num_samples,seed=args.seed)
    dataset = create_synthetic_madlib_dataset(genwrapper.tokenizer,template_str=args.story_str)

    generator = GenerateEvalBase(genwrapper = genwrapper, tokenizer=genwrapper.tokenizer)
    generated_responses = generator.generate_responses(dataset, genwrapper.sampling_repeat)
    
    # Save generations to a .pt file
    torch.save(generated_responses, f"{args.experiment_dir}/synthetic_responses.pt")

    print(f"Experiment completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
