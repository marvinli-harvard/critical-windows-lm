import os
from datetime import datetime
import pandas as pd 
import time
import json
import argparse
import torch
from datasets import load_dataset, Dataset
from typing import Dict
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import os
from tqdm import tqdm
from accelerate.utils import set_seed
from utils import *
from grader_utils import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Compute likelihood ratio between unaligned and aligned model to test for off
Example command line prompt

python run_likelihood_ratio_alignment.py --aligned_model meta-llama/Llama-3.1-8B-Instruct \
    --unaligned_model grimjim/Llama-3.1-8B-Instruct-abliterated_via_adapter   \
    --dataset Mechanistic-Anomaly-Detection/llama3-jailbreaks --split harmful_instructions_train --bs 16
"""


def main():
    parser = argparse.ArgumentParser()
    ## Model arguments
    parser.add_argument('--aligned_model', action="store", type=str, required=True, help="Name of aligned model")
    parser.add_argument('--unaligned_model', action="store", type=str, required=True, help="Name of unaligned model")

    parser.add_argument('--experiment_name', action="store", type=str, required=False, help='Optional experiment name.')
    
    ## Dataset arguments
    parser.add_argument('--dataset', action="store", type=str, required=True, help='Name of dataset.')
    parser.add_argument('--num_samples', action="store", type=int, required=False, help='Number of samples.')
    parser.add_argument('--split',  action="store", type=str, required=True, help='Split to compute over.')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size.')

    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=2243, help='Seed')
    args = parser.parse_args()
    
    start_time = time.time()

    ## Construct config file and save 
    if args.experiment_name is None:
        args.experiment_name = f"LikelihoodAligned_m1={args.aligned_model.replace('/','-')}_m2={args.unaligned_model.replace('/','-')}_dataset={args.dataset.replace('/','-').replace('@','-')}/split={args.split.replace('/','-')}/"        
        
    args.experiment_dir  = f"results/LikelihoodAligned/{args.experiment_name}/"
    args.date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    os.makedirs(os.path.dirname(args.experiment_dir), exist_ok=True)
    
    # Save to JSON
    with open(f"{args.experiment_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    ############################################################################################################################################################################################################################################
    ## Loading dataset and models 
    ############################################################################################################################################################################################################################################
    print("Loading pipeline and data")
    ## Load model and data
    set_seed(args.seed)
    aligned_pipe = load_model_pipeline(args.aligned_model)
    print("Finished loading aligned pipeline")

    if args.dataset == "Mechanistic-Anomaly-Detection/llama3-jailbreaks":
        dataset = load_dataset(args.dataset,split=args.split,trust_remote_code=True)
        dataset = dataset.map(lambda example: {
                            **example,
                            "full_text": example["prompt"]+example["completion"]
                            }
                        )
    else:
        assert False 
    dataset = dataset.shuffle(seed=args.seed)
    if args.num_samples:
        dataset=dataset.select(range(min(args.num_samples, len(dataset))))
    dataloader = DataLoader(dataset, batch_size = args.bs)
    print("Finished loading data")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    ############################################################################################################################################################################################################################################
    ## Computing losses
    ############################################################################################################################################################################################################################################
    aligned_losses = []
    print(f"Computing losses for aligned model {args.aligned_model}")
    for batch in tqdm(dataloader):
        inputs = aligned_pipe.tokenizer(batch["full_text"], return_tensors="pt", padding=True)
        aligned_losses.extend(list(compute_loss(aligned_pipe.model, inputs)))
    print(f"Finished computing losses for aligned model")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    torch.save(torch.tensor(aligned_losses), os.path.join(args.experiment_dir, "aligned_losses.pt"))
    
    del aligned_pipe, aligned_losses
    torch.cuda.empty_cache()

    print(f"Computing losses for unaligned model {args.unaligned_model}")
    unaligned_losses = [] 
    unaligned_pipe = load_model_pipeline(args.unaligned_model)
    for batch in tqdm(dataloader):
        inputs = unaligned_pipe.tokenizer(batch["full_text"], return_tensors="pt", padding=True)
        unaligned_losses.extend(list(compute_loss(unaligned_pipe.model, inputs)))
    
    print(f"Finished computing losses for unaligned model")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

    ############################################################################################################################################################################################################################################
    ## Save losses in args.experiment_dir
    ############################################################################################################################################################################################################################################
    
    torch.save(torch.tensor(unaligned_losses), os.path.join(args.experiment_dir, "unaligned_losses.pt"))


## I copied from the Pandora paper
def compute_loss(model, inputs): 
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"].long().to(device), 
                        attention_mask = inputs["attention_mask"].to(device))
        logits = outputs.logits.cpu()
        del outputs
    
    loss_fn = CrossEntropyLoss()
    input_ids_without_first_token = inputs["input_ids"][:, 1:].long()
    logits_without_last_token = logits[:, :-1, :]
    ans = []
    for i in range(len(logits_without_last_token)):
        first_unmasked = torch.where(inputs["attention_mask"][i]==1)[0].min()
        ce_loss = loss_fn(logits_without_last_token[i, first_unmasked:].float(), 
                          input_ids_without_first_token[i, first_unmasked:])
        ans.append(ce_loss)
    return torch.tensor(ans).cpu()

if __name__ == "__main__":
    main()
    
