import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import copy
import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from accelerate.utils import set_seed

from utils.utils import *
from utils.configuration import *
from utils.dataset_utils import *
from utils.generation_utils import *
from utils.grader_utils import *


from generation.GenerateEvalBase import *
from generation.GenerateEvalExamples import *

"""
Compute likelihood ratio between unaligned and aligned model to test for jailbreaks
Example command line prompt

python experiments/jailbreak/run_likelihood_ratio_jailbreak.py --aligned_model meta-llama/Llama-3.1-8B-Instruct \
    --unaligned_model grimjim/Llama-3.1-8B-Instruct-abliterated_via_adapter   \
    --num_samples 10
"""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def run_evaluation(generator, model, dataset_with_completion, prefix, experiment_dir):
    # Generate logprobs
    print(f"Generating logprobs for {prefix} dataset")
    logprobs = generator.generate_responses(dataset_with_completion, model.sampling_logprobs, return_logprobs=True)
    # Save results
    torch.save(logprobs, f"{experiment_dir}/{prefix}_logprobs.pt")
    

def load_model_and_generator(args, aligned):
    if aligned:
        model = load_all(model_id=args.aligned_model,
                                max_gen_length=1,
                                seed=args.seed,
                                log_probs=True)
    else:
        model = load_all(
            model_id=args.unaligned_model,
            generation_config_id=args.aligned_model,
            max_gen_length=1,
            seed=args.seed,
            log_probs=True
        )
    generator = GenerateEvalJailbreak(
        genwrapper=model,
        tokenizer=model.tokenizer
    )
    return model, generator 

def main():
    ####################################################################################################
    # SETUP
    ####################################################################################################
    parser = argparse.ArgumentParser()
    ## Model arguments
    parser.add_argument('--aligned_model', action="store", type=str, required=True, help="Name of aligned model")
    parser.add_argument('--unaligned_model', action="store", type=str, required=True, help="Name of jailbroken model")    

    ## Experiment config details
    parser.add_argument('--experiment_name', action="store", type=str, required=False, help='Optional experiment name.')
    
    ## Dataset argument
    parser.add_argument('--dataset', action="store", type=str, required=False, default=JAILBREAK_DATASET, help='Type of jailbreak dataset')
    parser.add_argument('--jailbreak_split',  action="store", nargs='+', type=str, required=False, 
                        default=["harmful_gcg","harmful_pair","harmful_autodan","harmful_msj",
                                 "harmful_human_mt","harmful_best_of_n","harmful_prefill","harmful_misc",], 
                        help='Jailbreak dataset')
    parser.add_argument('--benign_split', action="store", type=str, required=False, default="benign_instructions_test", help='Split for benign prompts.')
    parser.add_argument('--num_samples', action="store", type=int, required=False, default=None, help='Number of samples.')
    
    parser.add_argument('--tag', action="store", type=str, required=False, default=None, help='tag')
    parser.add_argument('--seed', action="store", type=int, required=False, default=DEFAULT_SEED, help='Seed')
    args = parser.parse_args()
    
    start_time = time.time()

    ## Construct config file and save 
    if args.experiment_name is None:
        args.experiment_name = f"JailbreakLikelihoodRatio_aligned={args.aligned_model.replace('/','-')}_unaligned={args.unaligned_model.replace('/','-')}_dataset={args.dataset.replace('/','-')}_num_samples={args.num_samples}"        
    args.experiment_dir  = f"results/JailbreakLikelihoodRatio/{args.experiment_name}/"
    args.date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    os.makedirs(os.path.dirname(args.experiment_dir), exist_ok=True)
    
    # Save to JSON
    with open(f"{args.experiment_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    ############################################################################################################################################################################################################################################
    ## Load aligned model and dataset
    ############################################################################################################################################################################################################################################
    set_seed(args.seed)
    
    
    jailbreak_dataset = create_prefill_dataset(
        tokenizer=None,
        dataset=args.dataset,
        char_step=None,
        num_samples=args.num_samples,
        filter_orig=False,
        jailbreak_suffix="",
        split=args.jailbreak_split
    )
    jailbreak_dataset_with_completion = copy.deepcopy(jailbreak_dataset).map(
        lambda example: {**example, "context": example["context"] + example["completion"]}
    )

    regular_dataset =  create_prefill_dataset(
        tokenizer=None,
        dataset=args.dataset,
        char_step=None,
        num_samples=args.num_samples,
        filter_orig=False,
        jailbreak_suffix="",
        split=args.benign_split
    )
    
    regular_dataset_with_completion = copy.deepcopy(regular_dataset).map(
        lambda example: {**example, "context": example["context"] + example["completion"]}
    )

    ################################################################################################################################################################
    ## Run evaluation for different models
    ################################################################################################################################################################

    for aligned_state in [True, False]:
        model, generator = load_model_and_generator(args, aligned=aligned_state)
        print(f"Running evaluation for aligned={aligned_state} model")
        label="aligned" if aligned_state else "unaligned"

        run_evaluation(
            generator=generator,
            model=model,
            dataset_with_completion=jailbreak_dataset_with_completion,
            prefix=f"{label}_jailbreak",
            experiment_dir=args.experiment_dir
        )

        run_evaluation(
            generator=generator,
            model=model,
            dataset_with_completion=regular_dataset_with_completion,
            prefix=f"{label}_benign",
            experiment_dir=args.experiment_dir
        )
        del model.model, generator
        del model
        torch.cuda.empty_cache()
        print(f"Experiment completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
