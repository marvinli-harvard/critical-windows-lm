"""
dataset_utils.py

Utilities for dataset handling and processing
"""

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from typing import Dict, List, Tuple
import re
import argparse
from utils.utils import * 


def get_dataset(args : argparse.Namespace) -> Dataset:
    """
    Load and preprocess a dataset based on the provided arguments.
    Args:
        args (Namespace): A namespace object containing the following attributes:
            - dataset (str): The name of the dataset to load.
            - split (str): The dataset split to load (e.g., 'train', 'test').
            - seed (int, optional): A seed for shuffling the dataset (used for 'cais/mmlu').
            - num_samples (int, optional): The number of samples to select from the dataset (used for 'cais/mmlu').
    Returns:
        Dataset: A Hugging Face dataset object with the specified preprocessing applied.
    Raises:
        AssertionError: If the specified dataset is not supported.
    """

    if args.dataset == "lucasmccabe/logiqa":
        dataset = load_dataset(args.dataset,split=args.split,trust_remote_code=True)
        dataset = dataset.map(lambda example: {
                    **example,
                    "problem": example["context"] + " " + example["query"] + "\n " \
                        + "\n ".join([f"{num_to_chr(i)} {opt}" for i, opt in enumerate(example["options"])]),
                    "formatted_answer": num_to_chr(example["correct_option"]),
                    },
                )
    elif args.dataset == "truthfulqa/truthful_qa":    
        dataset = load_dataset(args.dataset,"multiple_choice",split=args.split,trust_remote_code=True)
        def scramble_list(x, return_randperm=False):
            x = np.array(x)
            if return_randperm: 
                randperm = torch.randperm(len(x))
                return list(map(lambda x:str(x), list(x[randperm]))), randperm
            return list(map(lambda x:str(x), list(x[randperm])))
        def scramble_and_get_answer(example):
            scramble_list_choices, randperm = scramble_list(example["mc1_targets"]["choices"], return_randperm=True)    
            problem_str = example["question"] + " \n " \
                        + "\n ".join([f"{num_to_chr(i)} {opt}" for i, opt in enumerate(scramble_list_choices)])
            permutation = torch.tensor(example["mc1_targets"]["labels"])[randperm]
            i = torch.where(permutation==1)[0]

            return {"problem":problem_str, "formatted_answer":f"{num_to_chr(i)}"}
        dataset = dataset.map(lambda example: {**example, **scramble_and_get_answer(example)})
    elif args.dataset == "competition_math":
        dataset = load_dataset(args.dataset,split=args.split,trust_remote_code=True)
        def extract_comp_math_question(str, pattern=r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", default=None):
            # Applying the regex
            match = re.search(pattern, str)
            # Extract the content
            return match.group(1) if match else None
        dataset = dataset.map(lambda example:{**example,"formatted_answer":extract_comp_math_question(example["solution"])})
    elif args.dataset == "cais/mmlu":
        dataset = load_dataset("cais/mmlu", "all",split=args.split)
        dataset = dataset.map(lambda example: {
                    **example,
                    "problem": example["question"] + "\n " \
                        + "\n ".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(example["choices"])]),
                    "formatted_answer": num_to_chr(example["answer"])
                    }
                )
    elif args.dataset in ["allenai/ai2_arc@ARC-Challenge", "allenai/ai2_arc@ARC-Easy"]:
        dataset_name, dataset_config = args.dataset.split("@")
        dataset = load_dataset(dataset_name, dataset_config,split=args.split)
        dataset = dataset.map(lambda example: {
                    **example,
                    "problem": example["question"] + "\n " \
                        + "\n ".join([f"({label}) {text}" for label, text in zip(example["choices"]["label"], example["choices"]["text"])]),
                    "formatted_answer": add_parans(example["answerKey"])
                    },
                )
    elif args.dataset == "deepmind/aqua_rat":
        dataset = load_dataset(args.dataset,"raw", split=args.split)
        dataset = dataset.map(lambda example: {
                    **example,
                    "problem": example["question"] + "\n " \
                        + "\n ".join(example["options"]),
                    "formatted_answer": add_parans(example["correct"])
                    }
                )
    else:
        assert False, "Other datasets not supported"
    
    dataset = dataset.select_columns(["problem","formatted_answer"]).shuffle()
    if args.num_samples:
        dataset=dataset.select(range(min(args.num_samples, len(dataset))))
    return dataset

def create_dataframe(data :Dict[str, list], columns : List[str]) -> pd.DataFrame:
    return pd.DataFrame({col: [d[col] for d in data] for col in columns})

