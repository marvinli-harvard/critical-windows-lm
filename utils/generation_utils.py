"""
generation_utils.py

Some basic utilities for loaindg and generating with model
"""
from argparse import Namespace
from typing import Dict, List, Tuple
from tqdm import tqdm

import torch
import transformers

from vllm import LLM, SamplingParams

from prompt_generation.PromptGenerationBase import PromptGenerationBase
from utils.utils import *
from utils.configuration import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class GeneratorWrapper:
    """
    A wrapper class for a language model (LLM) that handles text generation with specified sampling parameters.
    Attributes:
        model (LLM): The language model to be used for text generation.
        tokenizer (transformers.AutoTokenizer): The tokenizer associated with the language model.
        sampling_first (SamplingParams): The sampling parameters to be used at the beginning of text generation.
        sampling_repeat (SamplingParams, optional): The sampling parameters to be used at for noised and denoised
    Methods:
        __init__(model, tokenizer, sampling_first, sampling_repeat):
            Initializes the GeneratorWrapper with the specified model, tokenizer, and sampling parameters.
    """
    def __init__(self, 
                 model : LLM, 
                 tokenizer : transformers.AutoTokenizer, 
                 sampling_first: SamplingParams,
                 sampling_repeat  : Optional[SamplingParams] = None):
        self.model = model 
        self.tokenizer = tokenizer
        self.sampling_first = sampling_first
        self.sampling_repeat  = sampling_repeat

## Load model through vllm
def load_all(model_id : str,
             max_gen_length : int,
             seed: int,
             num_per_noise : Optional[int] = None,
             ) -> GeneratorWrapper:
    """
    Load and configure the model, tokenizer, and sampling parameters for text generation.
    Args:
        model_id (str): The identifier of the pre-trained model to load.
        seed (int): The seed value for random number generation.
        max_gen_length (int): The maximum number of tokens to generate.
        num_per_noise (int, optional): The number of samples to generate per noise level.
    Returns:
        GeneratorWrapper: An object that wraps the model, tokenizer, and sampling parameters for text generation.
    """
    
    generation_config = transformers.GenerationConfig.from_pretrained(model_id)
    tokenizer = load_tokenizer(model_id)
    model = LLM(model=model_id,seed=seed)
    sampling_first = SamplingParams(temperature=generation_config.temperature, 
                                    top_p=generation_config.top_p,
                                    max_tokens=max_gen_length)
    if num_per_noise:
        sampling_repeat = SamplingParams(temperature=generation_config.temperature, 
                                        top_p=generation_config.top_p,
                                        max_tokens=max_gen_length,
                                        n=num_per_noise)
        return GeneratorWrapper(model=model, 
                                tokenizer=tokenizer,
                                sampling_first=sampling_first,
                                sampling_repeat=sampling_repeat)
    else:
        return GeneratorWrapper(model=model, 
                                tokenizer=tokenizer,
                                sampling_first=sampling_first)

def load_tokenizer(model_id : str) -> transformers.AutoTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_pipeline(model_id: str) -> transformers.pipeline:
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
        low_cpu_mem_usage=False,
    )
    return pipeline

## Generate tokens class
def generate_tokens(model : LLM, 
                    tokenizer : transformers.AutoTokenizer,
                    question_tokens : List[List[int]], 
                    prompt_info : Optional[PromptGenerationBase], 
                    sampling_params : SamplingParams) -> Tuple[List[int], List[str]]:
    generations = model.generate(prompt_token_ids=question_tokens, 
                                               sampling_params=sampling_params)
    outputs, responses = [], []
    for i in range(len(generations)):
        outp = []
        resp = []
        for j in range(len(generations[i].outputs)):
            output =  torch.tensor(generations[i].prompt_token_ids+list(generations[i].outputs[j].token_ids))
            if prompt_info:
                begin_index = torch.where(output==prompt_info.heading_to_tokens["begin_of_text"])[0][0]
                end_where = torch.where(output==prompt_info.heading_to_tokens["<|end_of_text|>"])[0]
                end_index = end_where[0] if len(end_where) > 0 else len(output)
                outp.append(output[begin_index:end_index])
                resp.append(tokenizer.decode(output[begin_index:end_index], skip_special_tokens=False))
            else:
                outp.append(output)
                resp.append(tokenizer.decode(output))
        outputs.append(outp)
        responses.append(resp)
    return outputs, responses


def get_raw_tokens_from_response(text :str ,tokenizer : transformers.AutoTokenizer)->torch.Tensor:
    return tokenizer(text, add_special_tokens=False)["input_ids"]
