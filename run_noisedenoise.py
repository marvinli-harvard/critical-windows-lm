import os
from datetime import datetime
import time
import json
import argparse
import numpy as np
import torch
from datasets import load_dataset, Dataset
from typing import Dict
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from accelerate.utils import set_seed
from utils import *
from LLAMANoiseDenoise import LLAMANoiseDenoise
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Example command line prompts
python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
    --dataset competition_math --split train --bs 1 --task math
python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
    --dataset lucasmccabe/logiqa --split train --bs 1 --task logic
python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
    --dataset truthfulqa/truthful_qa --split validation --bs 1 --task "true or false"
python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
    --dataset cais/mmlu --split auxiliary_train --bs 1 --task "multiple choice" --num_samples 10000
python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
    --dataset allenai/ai2_arc@ARC-Challenge \
    --split train --bs 1 --task science
run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
    --dataset allenai/ai2_arc@ARC-Easy \
    --split train --bs 1 --task science
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
    parser.add_argument('--cot_prompt', action="store", type=str, required=False, default = COT_PROMPT, help='Chain of thought prompt.')
    parser.add_argument('--system_prompt', action="store", type=str, required=False, default = SYSTEM_PROMPT, help='System prompt.')
    parser.add_argument('--percent_prompt', action="store", nargs='+', type=float, required=False, 
                        default=[0.1,0.3,0.5,0.7,0.9], help='List of percent prompts to include')
    parser.add_argument('--num_samples', action="store", type=int, required=False, help='Number of samples.')
    parser.add_argument('--max_gen_length', action="store", type=int, required=False, default=2048,help='Number of samples.')

    ## Dataset arguments
    parser.add_argument('--response_dataset', action="store", type=str, required=False, help='Location of dataset with responses.')
    parser.add_argument('--dataset', action="store", type=str, required=False, help='Name of dataset.')
    parser.add_argument('--split', action="store", type=str, required=True, help='Split of dataset.')
    
    parser.add_argument('--task', action="store", type=str, required=True, help='Type of dataset.')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size.')

    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=2243, help='Seed')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    args = parser.parse_args()
    
    start_time = time.time()

    ## Construct config file and save 
    args.cot_prompt = args.cot_prompt.replace("/TASK/", args.task)
    args.system_prompt = args.system_prompt.replace("/TASK/", args.task)
    if args.experiment_name is None:
        if args.dataset:               
            args.experiment_name = f"NoiseDenoise_model={args.model_id.replace('/','-')}_dataset={args.dataset.replace('/','-').replace('@','-')}_split={args.split}_nsamples={args.num_samples}"
        else:
            args.experiment_name = f"NoiseDenoise_model={args.model_id.replace('/','-')}_dataset=OLD_task={args.task}_split={args.split}_nsamples={args.num_samples}"
        
    args.experiment_dir  = f"results/NoiseDenoise/{args.experiment_name}/"
    args.date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    os.makedirs(os.path.dirname(args.experiment_dir), exist_ok=True)
    
    # Save to JSON
    with open(f"{args.experiment_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    ############################################################################################################################################################################################################################################
    ## Loading dataset, model, and prompt generator
    ############################################################################################################################################################################################################################################
    print("Loading dataset, model, and prompt generator")
    ## Load dataset
    if args.dataset:
        
        if args.dataset == "lucasmccabe/logiqa":
            dataset = load_dataset(args.dataset,split=args.split,trust_remote_code=True)
            dataset = dataset.map(lambda example: {
                        **example,
                        "problem": example["context"] + " " + example["query"] + "\n " \
                            + "\n ".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(example["options"])])
                        }
                    )
        elif args.dataset == "truthfulqa/truthful_qa":    
            dataset = load_dataset(args.dataset,"multiple_choice",split=args.split,trust_remote_code=True)
            def scramble_list(x):
                x = np.array(x)
                return list(map(lambda x:str(x), list(x[torch.randperm(len(x))])))
            
            dataset = dataset.map(lambda example: {**example, 
                                                   "problem": example["question"] + " \n " \
                                                   + "\n ".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(scramble_list(example["mc1_targets"]["choices"]))])
                                                   })
        elif args.dataset == "competition_math":
            dataset = load_dataset(args.dataset,split=args.split,trust_remote_code=True)
        elif args.dataset == "cais/mmlu":
            dataset = load_dataset("cais/mmlu", "all",split=args.split)
            dataset = dataset.shuffle(seed=args.seed).select(range(min(args.num_samples, len(dataset))))
            dataset = dataset.map(lambda example: {
                        **example,
                        "problem": example["question"] + "\n " \
                            + "\n ".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(example["choices"])])
                        }
                    )
        elif args.dataset in ["allenai/ai2_arc@ARC-Challenge", "allenai/ai2_arc@ARC-Easy"]:
            dataset_name, dataset_config = args.dataset.split("@")
            dataset = load_dataset(dataset_name, dataset_config,split=args.split)
            dataset = dataset.map(lambda example: {
                        **example,
                        "problem": example["question"] + "\n " \
                            + "\n ".join([f"({label}) {text}" for label, text in zip(example["choices"]["label"], example["choices"]["text"])])
                        }
                    )
        else:
            assert False, "Other datasets not supported"
        
        dataloader = DataLoader(dataset, batch_size = args.bs)
    elif args.response_dataset:
        first_responses   = torch.load(args.response_dataset)
    else:
        assert False, "Need to set either args.dataset or args.response_dataset"

    ## Load model
    set_seed(args.seed)
    pipeline = load_model_pipeline(args.model_id)
    if args.model_half:
        pipeline.half()

    ## Create prompt generator
    if args.model_id in ["meta-llama/Llama-3.1-8B-Instruct","meta-llama/Llama-3.1-70B-Instruct","meta-llama/Llama-3.1-405B-Instruct"]:
        prompt_gen = LLAMANoiseDenoise(cot_prompt=args.cot_prompt, 
                                       system_prompt=args.system_prompt,
                                       tokenizer=pipeline.tokenizer)
    else:
        assert False, "Other types of model_ids are not supported"
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    ############################################################################################################################################################################################################################################
    ## Get first response to model if necesssary
    ############################################################################################################################################################################################################################################
    if args.response_dataset is None:
        print("Generating basic responses for model")
        first_responses = []
        num_samples = 0
        for batch in tqdm(dataloader):
            if args.num_samples and num_samples >= args.num_samples:
                break
            
            # Convert all elements of the batch into valid question tokens
            inputs = [prompt_gen.get_question_tokens(item) for item in batch["problem"]]            
            max_inp_length = max([len(inp) for inp in inputs])
            inputs = torch.tensor(inputs)
            
            # Generate responses
            with torch.no_grad():
                inputs = inputs.to(device)
                outputs = pipeline.model.generate(inputs,max_new_tokens=args.max_gen_length)
        
            # Decode the responses
            responses = pipeline.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Store the information in a data structure
            curr_value = batch.copy()
            curr_value["question_tokens"] = inputs[0].cpu()
            curr_value["response_tokens"] = outputs[0][torch.where(outputs[0]==prompt_gen.heading_to_tokens["assistant"])[0][0]+2:].cpu()
            curr_value["full_response_tokens"] = outputs[0].cpu()
            curr_value["full_response_string"] = responses
            first_responses.append(curr_value.copy())
            num_samples += args.bs
        # Save responses to a JSON file
        args.response_dataset = os.path.join(args.experiment_dir, "dataset_with_gens.pt")
        torch.save(first_responses,args.response_dataset)

        print(f"Responses saved to {args.response_dataset}")
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    ############################################################################################################################################################################################################################################
    ## Now, running the noising-denoising experiments
    ############################################################################################################################################################################################################################################    
    response_with_noised_versions = []
    for batch in tqdm(first_responses):
        noised_responses = batch.copy()
        for stop_frac in args.percent_prompt:
            noised_responses[stop_frac] = {}
            
            ## Include noisy version of resposponses
            problem = noised_responses["problem"][0]
            if isinstance(problem, tuple):
                problem = problem[0]

            inputs = prompt_gen.get_noise_denoise_question(question=problem, 
                                                           response_tokens=noised_responses["response_tokens"].tolist(),
                                                           stop_frac=stop_frac)
            inputs = torch.tensor(inputs)
            noised_responses[stop_frac]["input_tokens"] = inputs
            noised_responses[stop_frac]["input_strings"] = pipeline.tokenizer.decode(inputs, 
                                                                                    skip_special_tokens=False)
            
            # Generate responses
            with torch.no_grad():
                inputs = inputs.to(device).reshape(-1,1).T
                add_length=max(args.max_gen_length-len(noised_responses["response_tokens"]),5)
                outputs = pipeline.model.generate(inputs, max_new_tokens=add_length)
            
            # Decode the responses
            responses = pipeline.tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Store the information in a data structure
            noised_responses[stop_frac]["full_response_tokens"] = outputs[0]
            noised_responses[stop_frac]["full_response_string"] = responses
        response_with_noised_versions.append(noised_responses)
    # Convert the response_with_noised_versions to a dataset and save to disk
    noise_denoise_path = os.path.join(args.experiment_dir, "response_with_noised_versions.pt")
    torch.save(response_with_noised_versions,noise_denoise_path)

    print(f"Final dataset with noised versions saved to {noise_denoise_path}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")




if __name__ == '__main__':
    main()

