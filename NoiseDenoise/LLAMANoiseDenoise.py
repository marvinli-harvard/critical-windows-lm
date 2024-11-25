import os
import re

import torch
import transformers
from typing import Dict, List
from NoiseDenoise.NoiseDenoiseBase import NoiseDenoiseBase

from utils.configuration import *
from utils.utils import *
from utils.dataset_utils import *
from utils.generation_utils import *
from utils.grader_utils import *
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class LLAMANoiseDenoise(NoiseDenoiseBase):
    def __init__(self, 
                 system_prompt: str, 
                 cot_prompt: str, 
                 tokenizer: transformers.PreTrainedTokenizer,
                 clarify_choice_str : str):
        self.system_prompt = system_prompt
        self.cot_prompt = cot_prompt
        self.tokenizer = tokenizer
        self.heading_to_tokens = {
            "system":9125,
            "user":882,
            "assistant":78191,
            "eot_id":128009,
            "begin_of_text":128000,
            "<|end_of_text|>":128001
        }
        self.clarify_choice_str = clarify_choice_str
    
    def get_question_tokens(self, question:str, include_stepbystep : bool = True) -> List[int]:
        system_heading           = self.return_heading_list_llama("system",
                                                             start=True)
        system_tokens            = get_raw_tokens_from_response(self.system_prompt,
                                                                self.tokenizer)
        user_heading             = self.return_heading_list_llama("user",
                                                             start=False)
        user_tokens              = get_raw_tokens_from_response(question
                                                                +"\n\n\n"
                                                                +self.cot_prompt,
                                                                self.tokenizer)
        assistant_heading        = self.return_heading_list_llama("assistant",
                                                             start=False)
        if include_stepbystep:
            assistant_heading += get_raw_tokens_from_response("Let's think step by step: ",
                                                              self.tokenizer)
        return system_heading + system_tokens + [self.heading_to_tokens["eot_id"]] \
                + user_heading + user_tokens + [self.heading_to_tokens["eot_id"]] \
                + assistant_heading 
                        
    
    def get_noise_denoise_question(self, question:str, response_tokens, stop_frac:float=0.2) -> List[int]:
        return self.get_question_tokens(question,False) \
                + response_tokens[:int(stop_frac * len(response_tokens))]

    def return_heading_list_llama(self, type :str, start=False) -> List[int]:
        """
        Generates a list of token IDs representing the heading structure for LLAMA.
        Args:
            type (str): The type of heading to be converted to tokens.
            start (bool, optional): If True, includes the beginning of text token. Defaults to False.
        Returns:
            list: A list of token IDs representing the heading structure.
        """
        if start:
            return [
                    128000,  # <|begin_of_text|>
                    128006,  # <|start_header_id|>
                    self.heading_to_tokens[type],
                    128007,  # <|end_header_id|>
                    271,     # "\n\n"
            ]
        else:
            return [
                    128006,  # <|start_header_id|>
                    self.heading_to_tokens[type],
                    128007,  # <|end_header_id|>
                    271,     # "\n\n"
            ]

    def complete_with_answer(self, 
                            existing_tokens : List[int]) -> List[int]:
        complete_tokens     = existing_tokens
        complete_tokens     += self.return_heading_list_llama("user",start=False)
        complete_tokens     += get_raw_tokens_from_response(self.clarify_choice_str,self.tokenizer) 
        complete_tokens     += [self.heading_to_tokens["eot_id"]]
        complete_tokens     += self.return_heading_list_llama("assistant",start=False)
        return complete_tokens
    