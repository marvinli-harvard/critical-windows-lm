from typing import Dict, List, Optional

import transformers
from vllm import SamplingParams

from utils.configuration import *
from utils.utils import *
from utils.dataset_utils import *
from utils.generation_utils import *
from utils.grader_utils import *

from prompt_generation.PromptGenerationBase import PromptGenerationBase
from generation.GenerationBase import GenerationBase

class GenerateEvalBase(GenerationBase):
    def __init__(self, **kwargs
                 ):        
        super().__init__(**kwargs)
    
    def generate_responses(self, 
                           dataset : Dataset,
                         sampling_params : SamplingParams,
                         return_logprobs : bool = False)->List[Dict]:
        
        input_tokens = [self.tokenizer.encode(dict_entry["context"][0]) for dict_entry in dataset.iter(batch_size=1)]
        if return_logprobs:
            outputs, responses, logprobs = self.generate(input_tokens,sampling_params, return_logprobs=return_logprobs)
        else:
            outputs, responses = self.generate(input_tokens,sampling_params, return_logprobs=return_logprobs)
    
        final_answers = []
        for i, val in tqdm(list(enumerate(dataset.iter(batch_size=1)))):
            for j in range(len(responses[i])):
                curr_value = {k:v[0] for k,v in val.items()}
                curr_value["input_tokens"]    = input_tokens[i]
                curr_value["response_tokens"] = outputs[i][j]
                curr_value["response_string"] = responses[i][j]
                curr_value["no"] = j
                if return_logprobs:
                    curr_value["prompt_logprobs"] = logprobs[i][j]["prompt_logprobs"]
                    curr_value["gen_logprobs"] = logprobs[i][j]["gen_logprobs"]
                final_answers.append(curr_value)    

        return final_answers
    

    def grade(self, model_answers : List[Dict], **kwargs) -> List[Dict]:
        raise NotImplementedError("Need to implement grader type")
    

