from typing import Dict, List, Optional

import transformers
from vllm import SamplingParams

from utils.configuration import *
from utils.utils import *
from utils.dataset_utils import *
from utils.generation_utils import *
from utils.grader_utils import *
from prompt_generation.PromptGenerationBase import PromptGenerationBase


class GenerationBase:
    def __init__(self, 
                 genwrapper : GeneratorWrapper,
                 tokenizer : transformers.AutoTokenizer,
                 prompt_gen : Optional[PromptGenerationBase] = None,
                 **kwargs
                 ):
        self.genwrapper = genwrapper
        self.tokenizer = tokenizer 
        self.prompt_gen = prompt_gen
    
    def generate(self, input_tokens : List[List[int]],
                 sampling_params : SamplingParams,
                 return_logprobs = False):
        return generate_tokens(model=self.genwrapper.model, 
                tokenizer=self.tokenizer, 
                question_tokens=input_tokens, 
                prompt_info=self.prompt_gen, 
                sampling_params = sampling_params,
                return_logprobs = return_logprobs)
    
    
