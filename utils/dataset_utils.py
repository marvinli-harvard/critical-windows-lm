"""
dataset_utils.py

Utilities for dataset handling and processing
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
import re
import argparse
import itertools
from enum import Enum

from datasets import Dataset, load_dataset

from utils.utils import * 
from utils.grader_utils import *
from utils.generation_utils import * 


# Basic data manipulations
def create_dataframe(data :Dict[str, list], columns : List[str]) -> pd.DataFrame:
    return pd.DataFrame({col: [d[col] for d in data] for col in columns})

# Get Q&A dataset for COT experiments
def get_qa_dataset(dataset : str, 
                   split : str,
                   num_samples : Optional[int]) -> Dataset:
    """
    Load and preprocess QA dataset based on the provided arguments.
    Args:
        dataset (str) : The name of the dataset to load.
        split (str)   : The dataset split to load.
        num_samples (int, optional): The number of samples to select from the dataset.
    Returns:
        Dataset: A Hugging Face dataset object with the specified preprocessing applied.
    Raises:
        AssertionError: If the specified dataset is not supported.
    """

    if dataset == "lucasmccabe/logiqa":
        dataset = load_dataset(dataset,split=split,trust_remote_code=True)
        dataset = dataset.map(lambda example: {
                    **example,
                    "problem": example["context"] + " " + example["query"] + "\n " \
                        + "\n ".join([f"{num_to_chr(i)} {opt}" for i, opt in enumerate(example["options"])]),
                    "formatted_answer": num_to_chr(example["correct_option"]),
                    },
                )
    elif dataset == "truthfulqa/truthful_qa":    
        dataset = load_dataset(dataset,"multiple_choice",split=split,trust_remote_code=True)
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
    elif dataset == "competition_math":
        dataset = load_dataset(dataset,split=split,trust_remote_code=True)
        dataset = dataset.map(lambda example:{**example,"formatted_answer":extract_comp_math_question(example["solution"])})
    elif dataset == "cais/mmlu":
        dataset = load_dataset("cais/mmlu", "all",split=split)
        dataset = dataset.map(lambda example: {
                    **example,
                    "problem": example["question"] + "\n " \
                        + "\n ".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(example["choices"])]),
                    "formatted_answer": num_to_chr(example["answer"])
                    }
                )
    elif dataset in ["allenai/ai2_arc@ARC-Challenge", "allenai/ai2_arc@ARC-Easy"]:
        dataset_name, dataset_config = dataset.split("@")
        dataset = load_dataset(dataset_name, dataset_config,split=split)
        dataset = dataset.map(lambda example: {
                    **example,
                    "problem": example["question"] + "\n " \
                        + "\n ".join([f"({label}) {text}" for label, text in zip(example["choices"]["label"], example["choices"]["text"])]),
                    "formatted_answer": add_parans(example["answerKey"])
                    },
                )
    elif dataset == "deepmind/aqua_rat":
        dataset = load_dataset(dataset,"raw", split=split)
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
    if num_samples:
        dataset=dataset.select(range(min(num_samples, len(dataset))))
    return dataset

# Get structured output datasets
def create_structured_form_dataset(tokenizer : transformers.PreTrainedTokenizer,    
                                    template_str : str,
                                    starts_with : str = "\n\n1. "
                                    )->Dataset:
    
    
    return Dataset.from_dict({"context":[prompt_to_prefix(template_str, starts_with, tokenizer)]})
    
def remove_duplicates_by_prompt_text(dataset,col):
    seen_prompt_texts = set()
    def filter_unique(example):
        prompt_text = example[col]
        if prompt_text in seen_prompt_texts:
            return False
        seen_prompt_texts.add(prompt_text)
        return True

    dataset = dataset.filter(filter_unique)
    return dataset

# Get  jailbreak  with 
def create_prefill_dataset(
                           tokenizer   :  transformers.PreTrainedTokenizer,    
                           dataset : str = JAILBREAK_DATASET,
                           jailbreak_suffix : str = JAILBREAK_PREFILL,
                           char_step : Optional[int] = 4,
                           num_samples : Optional[int] = None,
                           filter_orig : bool = True
                            ) -> Dataset:
    """
    Creates a jailbreak dataset where a prefix is appended to a jailbreak dataset
    
    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to be used for encoding the text.
        harmful_request_dataset (str): Path to the dataset containing harmful requests.
        prefill_attack (str, optional): Prefix to be appended to each request. Defaults to PREFILL_ATTACK.
    Returns:
        Dataset: A Hugging Face dataset object
    Raises:
        AssertionError: If the specified dataset is not supported.
    """    
    
    if dataset == "sevdeawesome/jailbreak_success":
        dataset = load_dataset(dataset,split="train").shuffle()
        if filter_orig:
            dataset = dataset.filter(lambda example: example["jailbreak_prompt_name"] == "original_prompt")        
        else:
            dataset = dataset.filter(lambda example: example["jailbreak_prompt_name"] in ["prefix_injection_hello", 
                                                                                          "original_prompt", "wikipedia", 
                                                                                          "style_injection_short",
                                                                                            "autodan"])
        dataset = remove_duplicates_by_prompt_text(dataset,col="jailbreak_prompt_text")
        def example_to_stuff(example, prefix):
            return {"toxic_type": example["prompt_name"].replace("_", " "),
                    "context": prompt_to_prefix(example["jailbreak_prompt_text"], prefix, tokenizer),
                    "length":len(prefix)}
    elif dataset == "hakurei/open-instruct-v1":
        dataset = load_dataset("hakurei/open-instruct-v1",split="train").shuffle()
        dataset = dataset.filter(lambda example: example["input"] == "")        
        dataset = remove_duplicates_by_prompt_text(dataset,col="instruction")
        def example_to_stuff(example, prefix):
            return {"context": prompt_to_prefix(example["instruction"], prefix, tokenizer),"length":len(prefix)}        
    else:
        raise ValueError("Does not support this type of adversarial dataset")
    
    prefill_attackes = generate_prefill_attacks(jailbreak_suffix, char_step)
    datasets = []
    for prefix in prefill_attackes:
        dataset_with_prefix = dataset.map(lambda example: example_to_stuff(example, prefix))
        datasets.append(dataset_with_prefix)
    dataset = Dataset.from_dict({key: sum([d[key] for d in datasets], []) for key in datasets[0].column_names})
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))        

    return dataset 


def create_repetition_dataset(tokenizer : transformers.PreTrainedTokenizer,
                               base_prompt : str = REPEAT_WORD_USER_PROMPT,
                               repeat_words : List[str] = REPEAT_WORD_LIST,
                               repeat_times : List[int] = REPEAT_WORDS_TIMES
                               ) -> Dataset:
    """
    Creates a dataset where the inputs are generated by repeating specified words a given number of times.
    
    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer
        base_prompt (str, optional): The base prompt to be used for generating the dataset. Defaults to REPEAT_WORD_USER_PROMPT.
        repeat_words (List[str], optional): A list of words to be repeated in the prompts. Defaults to REPEAT_WORD_LIST.
        num_repeat (List[int], optional): A list of integers specifying how many times each word in repeat_words should be repeated. Defaults to REPEAT_WORDS_TIMES.        
    Returns:
        Dataset: A Hugging Face dataset object with the specified preprocessing applied.
    """    
    list_of_contexts = []
    list_of_answers  = []
    list_of_times  = []
    for word, times in itertools.product(repeat_words, repeat_times):
        list_of_contexts.append(prompt_to_prefix(base_prompt, f"{word} "*times, tokenizer))
        list_of_answers.append(word)
        list_of_times.append(times)
    
    dataset = Dataset.from_dict({"context": list_of_contexts,
                                 "repeated_word":list_of_answers,
                                 "times":list_of_times})
    return dataset

def prompt_to_prefix(prompt: str, 
                     prefix : str, 
                     tokenizer:  transformers.PreTrainedTokenizer):
    header = tokenizer.decode(tokenizer.apply_chat_template([{"role":"user","content":prompt}]))
    return header+"<|start_header_id|>assistant<|end_header_id|>\n\n"+prefix 

def generate_prefill_attacks(prefill_attack, char_step):
    if char_step:
        return [prefill_attack[:i] for i in range(char_step, len(prefill_attack) + char_step, char_step)]
    else:
        return [prefill_attack]
    
