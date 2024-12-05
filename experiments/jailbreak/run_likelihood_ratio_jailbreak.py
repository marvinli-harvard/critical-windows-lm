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



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Compute likelihood ratio between unaligned and aligned model to test for off
Example command line prompt

python experiments/jailbreak/run_likelihood_ratio_jailbreak.py --aligned_model meta-llama/Llama-3.1-8B-Instruct \
    --unaligned_model grimjim/Llama-3.1-8B-Instruct-abliterated_via_adapter   \
    --num_samples 10000 --num_repeats 100
"""


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
    ## Dataset arguments
    parser.add_argument('--dataset', action="store", type=str, required=False, default=JAILBREAK_DATASET, help='Type of jailbreak dataset')
    parser.add_argument('--num_samples', action="store", type=int, required=False, default=None, help='Number of samples.')
    parser.add_argument('--num_repeats', action="store", type=int, required=False, default=1, help='Number of repeats to evaluate jailbreaks.')
    
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=DEFAULT_SEED, help='Seed')
    args = parser.parse_args()
    
    start_time = time.time()

    ## Construct config file and save 
    if args.experiment_name is None:
        args.experiment_name = f"JailbreakLikelihoodRatio_aligned_model={args.aligned_model.replace('/','-')}_dataset={args.dataset.replace('/','-')}_num_samples={args.num_samples}"        
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
    ## Construct jailbroken dataset and load model
    jailbreak_dataset = create_prefill_jailbreak_dataset(tokenizer=aligned_model.tokenizer,
                                jailbreak_dataset=args.dataset,
                                char_step=None,
                                num_samples=args.num_samples,
                                filter_orig=False,
                                jailbreak_suffix="")
    
    jailbreak_dataset_with_prefill = copy.deepcopy(jailbreak_dataset)
    jailbreak_dataset_with_prefill = jailbreak_dataset_with_prefill.map(
        lambda example: {**example, "context": example["context"] + args.evaluate_logprob_str})
    aligned_generator = GenerateEvalJailbreak(genwrapper = aligned_model, 
                                      tokenizer=aligned_model.tokenizer)
    
    ############################################################################################################################################################################################################################################
    ## Run evaluation and grade each answer
    ############################################################################################################################################################################################################################################
    print("Generating logprobs for aligned model")
    aligned_jailbreak_logprobs = aligned_generator.generate_responses(jailbreak_dataset_with_prefill, aligned_model.sampling_logprobs, return_logprobs=True)
    
    print("Generating responses for aligned model")
    aligned_jailbreak_resp = aligned_generator.generate_responses(jailbreak_dataset, aligned_model.sampling_repeat)

    del aligned_model.model
    torch.cuda.empty_cache()

    print("Grading resposnes for aligned model")
    aligned_jailbreak_graded = aligned_generator.grade(aligned_jailbreak_resp)

    unaligned_model = load_all(model_id=args.unaligned_model,
                               generation_config_id=args.aligned_model,
                               max_gen_length=args.max_gen_length,
                             seed=args.seed,
                             num_per_noise=args.num_repeats,
                             log_probs=True)
    unaligned_generator = GenerateEvalJailbreak(genwrapper = unaligned_model, 
                                      tokenizer=unaligned_model.tokenizer)
    print("Generating logprobs for unaligned model")
    unaligned_jailbreak_logprobs = unaligned_generator.generate_responses(jailbreak_dataset_with_prefill, unaligned_model.sampling_logprobs, return_logprobs=True)
    
    print("Generating responses for unaligned model")
    unaligned_jailbreak_resp = unaligned_generator.generate_responses(jailbreak_dataset, unaligned_model.sampling_repeat)

    del unaligned_model.model
    print("Grading resposnes for unaligned model")
    unaligned_jailbreak_graded = unaligned_generator.grade(unaligned_jailbreak_resp)

    # Save logprobs and graded responses to .pt files
    torch.save(aligned_jailbreak_logprobs, f"{args.experiment_dir}/aligned_jailbreak_logprobs.pt")
    torch.save(aligned_jailbreak_graded, f"{args.experiment_dir}/aligned_jailbreak_graded.pt")
    torch.save(unaligned_jailbreak_logprobs, f"{args.experiment_dir}/unaligned_jailbreak_logprobs.pt")
    torch.save(unaligned_jailbreak_graded, f"{args.experiment_dir}/unaligned_jailbreak_graded.pt")
    print(f"Experiment completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
    
