from typing import Dict, List, Optional

import transformers
from vllm import SamplingParams

from utils.configuration import *
from utils.utils import *
from utils.dataset_utils import *
from utils.generation_utils import *
from utils.grader_utils import *
from generation.GenerateEvalBase import GenerateEvalBase

class GenerateEvalJailbreak(GenerateEvalBase):
    def __init__(self, **kwargs):        
        super().__init__(**kwargs)
    
    def grade(self, model_answers : List[Dict]) -> List[Dict]:
        self.jailbreakclassifier = JailBreakClassifierWrapper()
        behaviors = [ans["toxic_type"] for ans in model_answers]
        answers = [ans["response_string"].split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]  for ans in model_answers]    
        grader_answers = self.jailbreakclassifier.grade_answers(behaviors, answers)
        response = []
        
        for i in range(len(model_answers)):
            curr = model_answers[i].copy()
            curr["grader_answer"] = grader_answers[i]
            response.append(curr)
        
        del self.jailbreakclassifier.cls
        torch.cuda.empty_cache()
        
        return response 
    
class GenerateEvalRepeat(GenerateEvalBase):
    def __init__(self, **kwargs):        
        super().__init__(**kwargs)
    
    def grade(self, model_answers : List[Dict]) -> List[Dict]:
        response = []
        for i in range(len(model_answers)):
            curr = model_answers[i].copy()
            curr["grader_answer"] =  curr["response_string"].count(curr["repeated_word"])
            response.append(curr)
        return response

