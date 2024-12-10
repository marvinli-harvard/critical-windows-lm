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
Compute likelihood ratio between unaligned and aligned model to test for off distribution behavior
Example command line prompt

python experiments/jailbreak/run_likelihood_ratio_jailbreak.py --aligned_model meta-llama/Llama-3.1-8B-Instruct \
    --unaligned_model grimjim/Llama-3.1-8B-Instruct-abliterated_via_adapter   \
    --num_samples 10 --num_repeats 10
"""

def run_evaluation(generator, model, dataset, dataset_with_prefill, prefix, experiment_dir):
    # Generate logprobs
    print(f"Generating logprobs for {prefix} dataset")
    logprobs = generator.generate_responses(dataset_with_prefill, model.sampling_logprobs, return_logprobs=True)
    
    # Generate responses
    print(f"Generating responses for {prefix} dataset")
    responses = generator.generate_responses(dataset, model.sampling_repeat)
    
    del model.model
    torch.cuda.empty_cache()

    # Grade responses
    print(f"Grading responses for {prefix} dataset")
    graded = generator.grade(responses)
    
    # Save results
    torch.save(logprobs, f"{experiment_dir}/{prefix}_logprobs.pt")
    torch.save(graded, f"{experiment_dir}/{prefix}_graded.pt")

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
    parser.add_argument('--max_gen_length', action="store", type=int, required=False, default=JAILBREAK_MAX_LEN,help='Maximum generation length')
    parser.add_argument('--evaluate_logprob_str', action="store", type=str, required=False, default=JAILBREAK_PREFILL,help='Default log prob statement to compute')
    
    ## Dataset argument
    parser.add_argument('--jailbreak_dataset', action="store", type=str, required=False, default=JAILBREAK_DATASET, help='Type of jailbreak dataset')
    parser.add_argument('--num_samples', action="store", type=int, required=False, default=None, help='Number of samples.')
    parser.add_argument('--num_repeats', action="store", type=int, required=False, default=1, help='Number of repeats to evaluate jailbreaks.')
    
    parser.add_argument('--seed', action="store", type=int, required=False, default=DEFAULT_SEED, help='Seed')
    args = parser.parse_args()
    
    start_time = time.time()

    ## Construct config file and save 
    if args.experiment_name is None:
        args.experiment_name = f"JailbreakLikelihoodRatio_aligned_model={args.aligned_model.replace('/','-')}_dataset={args.jailbreak_dataset.replace('/','-')}_num_samples={args.num_samples}"        
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
    
    aligned_model = load_all(model_id=args.aligned_model,
                             max_gen_length=args.max_gen_length,
                             seed=args.seed,
                             num_per_noise=args.num_repeats,
                             log_probs=True)
    jailbreak_dataset = create_prefill_dataset(
        tokenizer=aligned_model.tokenizer,
        dataset=args.jailbreak_dataset,
        char_step=None,
        num_samples=args.num_samples,
        filter_orig=False,
        jailbreak_suffix=""
    )

    # Add evaluation logprob string to the dataset
    jailbreak_dataset_with_prefill = copy.deepcopy(jailbreak_dataset)
    jailbreak_dataset_with_prefill = jailbreak_dataset_with_prefill.map(
        lambda example: {**example, "context": example["context"] + args.evaluate_logprob_str}
    )

    aligned_generator = GenerateEvalJailbreak(
        genwrapper=aligned_model,
        tokenizer=aligned_model.tokenizer
    )
    ################################################################################################################################################################
    ## Run evaluation for aligned model
    ################################################################################################################################################################
    print("Running evaluation for aligned model")
    run_evaluation(
        generator=aligned_generator,
        model=aligned_model,
        dataset=jailbreak_dataset,
        dataset_with_prefill=jailbreak_dataset_with_prefill,
        prefix="aligned_jailbreak",
        experiment_dir=args.experiment_dir
    )

    ################################################################################################################################################################
    ## Run evaluation for unaligned model
    ################################################################################################################################################################
    unaligned_model = load_all(
        model_id=args.unaligned_model,
        generation_config_id=args.aligned_model,
        max_gen_length=args.max_gen_length,
        seed=args.seed,
        num_per_noise=args.num_repeats,
        log_probs=True
    )

    unaligned_generator = GenerateEvalJailbreak(
        genwrapper=unaligned_model, 
        tokenizer=unaligned_model.tokenizer
    )

    print("Running evaluation for unaligned model")
    run_evaluation(
        generator=unaligned_generator,
        model=unaligned_model,
        dataset=jailbreak_dataset,
        dataset_with_prefill=jailbreak_dataset_with_prefill,
        prefix="unaligned_jailbreak",
        experiment_dir=args.experiment_dir
    )

    print(f"Experiment completed in {time.time() - start_time:.2f} seconds.")




if __name__ == "__main__":
    main()
