from typing import Dict, List, Optional

import transformers
from vllm import SamplingParams

from src.cwlm.utils.configuration import *
from src.cwlm.utils.utils import *
from src.cwlm.utils.dataset_utils import *
from src.cwlm.utils.generation_utils import *
from src.cwlm.utils.grader_utils import *
from src.cwlm.prompt_generation.PromptGenerationBase import PromptGenerationBase


class GenerationBase:
    def __init__(self, 
                 model : GeneratorWrapper,
                 tokenizer : transformers.AutoTokenizer,
                 prompt_gen : Optional[PromptGenerationBase] = None,
                 **kwargs
                 ):
        self.model = model 
        self.tokenizer = tokenizer 
        self.prompt_gen = prompt_gen

    def update_sampling_params(self, sampling_params_new : SamplingParams):
        self.prompt_gen = sampling_params_new

    def update_prompt_gen(self, prompt_gen_new : PromptGenerationBase):
        self.prompt_gen = prompt_gen_new
    
    def generate(self, input_tokens : List[List[int]],
                 sampling_params : SamplingParams):
        token_outputs, string_outputs = generate_tokens(model=self.model, 
                                             tokenizer=self.tokenizer, 
                                             question_tokens=input_tokens, 
                                             prompt_info=self.prompt_gen, 
                                             sampling_params = sampling_params)
        return token_outputs, string_outputs
    
