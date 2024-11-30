from typing import Dict, List, Optional

import transformers

from datasets import Dataset, load_dataset
from vllm import SamplingParams

from src.cwlm.utils.configuration import *
from src.cwlm.utils.utils import *
from src.cwlm.utils.dataset_utils import *
from src.cwlm.utils.generation_utils import *
from src.cwlm.utils.grader_utils import *

from src.cwlm.prompt_generation.PromptGenerationBase import PromptGenerationBase
from src.cwlm.generation.GenerationBase import GenerationBase

class QAGenerateNoiseDenoise(GenerationBase):
    def __init__(self, dataset : Dataset, answer_type  : str, **kwargs):
        self.dataset = dataset 
        self.answer_type = answer_type
        super.__init_(**kwargs)

    def generate_basic(self) -> List[Dict]:
        self.first_responses = []
        question_tokens = [self.prompt_gen.get_question_tokens(item) for item in self.dataset["problem"]]
        outputs, responses = self.generate(question_tokens, self.model.sampling_first)
        for i, val in tqdm(list(enumerate(self.dataset.iter(batch_size=1)))):
            curr_value = {k:v[0] for k,v in val.items()}
            asst_index = torch.where(outputs[i][0]==self.prompt_gen.heading_to_tokens["assistant"])[0][0]+2
            
            curr_value["question_tokens"] = question_tokens[i]
            curr_value["orig_tokens"] = outputs[i][0]
            curr_value["orig_string"] = responses[i][0]
            curr_value["asst_tokens"] = outputs[i][0][asst_index:]
            self.first_responses.append(curr_value.copy())
        return self.first_responses
    
    def generate_noised_denoised_questions(self, percent_prompt : List[float]) -> List[Dict]:
        noised_questions_tokens = []
        for gens in tqdm(self.first_responses):
            for stop_frac in percent_prompt:
                entry = gens.copy()
                entry["stop_frac"] = stop_frac
                inputs = self.prompt_gen.get_noise_denoise_question(question=entry["problem"], 
                                                            response_tokens=entry["asst_tokens"].tolist(),
                                                            stop_frac=stop_frac)
                entry["no_deno_input_tokens"] = list(inputs)
                entry["no_deno_input_string"] = self.model.tokenizer.decode(inputs, skip_special_tokens=False)
                noised_questions_tokens.append(entry)
        return noised_questions_tokens
    
    def generate_noised_denoised(self, percent_prompt : List[float]):
        noised_questions_tokens = self.generate_noised_denoised_questions(percent_prompt=percent_prompt)
        
        ## Now run generation
        question_tokens = [item["no_deno_input_tokens"] for item in noised_questions_tokens]
        outputs, responses = self.generate(question_tokens, self.model.sampling_repeat)
        self.noised_denoised_results = []
        for i in tqdm(range(len(outputs))):    
            batch_output   = outputs[i]
            batch_response = responses[i]
            for j in range(len(batch_output)):
                curr_value = noised_questions_tokens[i].copy()
                curr_value["no_deno_output_tokens"]     = batch_output[j]
                curr_value["no_deno_output_strings"]    = batch_response[j]
                curr_value["no"]                        = j
                self.noised_denoised_results.append(curr_value)
        return self.noised_denoised_results
    
    def generate_orig_stump(self):
        self.existing_orig_responses  = {}
        self.existing_stump_responses = {}
        for example in self.noised_denoised_results:
            if example["problem"] not in self.existing_orig_responses:
                self.existing_orig_responses[example["problem"]] = \
                    self.prompt_gen.complete_with_answer(example["orig_tokens"].tolist())
            if tuple(example["no_deno_input_tokens"]) not in self.existing_stump_responses:
                self.existing_stump_responses[tuple(example["no_deno_input_tokens"])] = \
                    self.prompt_gen.complete_with_answer(example["no_deno_input_tokens"] + [self.prompt_gen.heading_to_tokens["eot_id"]])
        return self.existing_orig_responses, self.existing_stump_responses

    def generate_original_answer(self):
        print("Computing answers of original responses")
        orig_responses   = list(self.existing_orig_responses.items())
        orig_tokens  = [item[1] for item in orig_responses] 
        
        self.orig_answers     = {} 
        outputs_orig, response_orig =  self.generate(orig_tokens, self.model.sampling_first) 
        for i in range(len(orig_responses)):
            new_tokens = outputs_orig[i][0][len(orig_tokens[i]):-1]
            new_string = self.model.tokenizer.decode(new_tokens)
            self.orig_answers[orig_responses[i][0]] = {
                "orig_tokens_ans": outputs_orig[i][0],
                "orig_string_ans": response_orig[i][0],
                "orig_ans_tokens": new_tokens,
                "orig_ans_string": new_string,
                "orig_ans_format": extract_answer(new_string, self.answer_type),
            }
        

    def generate_stump_answers(self):
        stump_responses = list(self.existing_stump_responses.items())    
        stump_tokens  = [item[1] for item in stump_responses]    
        self.stump_answers     = {} 
        outputs_stump, response_stump = self.generate(stump_tokens, self.model.sampling_first) 
        for i in range(len(stump_responses)):
            new_tokens = outputs_stump[i][0][len(stump_tokens[i]):-1]
            new_string = self.model.tokenizer.decode(new_tokens)
            self.stump_answers[stump_responses[i][0]] = {
                "stump_tokens_ans": outputs_stump[i][0],
                "stump_string_ans": response_stump[i][0],
                "stump_ans_tokens": new_tokens,
                "stump_ans_string": new_string,
                "stump_ans_format": extract_answer(new_string, args.answer_type),
            }
        return self.stump_answers

    def generate_noised_denoised_answer(self):
        self.results = []
        noised_denoised_results_tokens = [self.prompt_gen.complete_with_answer(b["no_deno_output_tokens"].tolist()) 
                            for b in self.noised_denoised_results]
        outputs_noise, response_noise = self.generate(noised_denoised_results_tokens, self.model.sampling_first) 
        for i in tqdm(range(len(self.noised_denoised_results))):
            curr_results = self.noised_denoised_results[i].copy()
            curr_results["no_deno_tokens_ans"] = outputs_noise[i][0]
            curr_results["no_deno_string_ans"] = response_noise[i][0]
            new_tokens = outputs_noise[i][0][len(noised_denoised_results_tokens[i]):-1]
            new_string = self.model.tokenizer.decode(new_tokens)

            curr_results["no_deno_ans_tokens"]  = new_tokens
            curr_results["no_deno_ans_string"]  = new_string
            curr_results["no_deno_ans_format"]  = extract_answer(new_string, self.answer_type)
            curr_results.update(self.orig_answers[curr_results["problem"]].copy())
            curr_results.update(self.stump_answers[tuple(curr_results["no_deno_input_tokens"])].copy())
            
            curr_results["orig_is_right"]   = compare_answers(curr_results["orig_ans_format"],
                                                                curr_results["formatted_answer"],
                                                                self.answer_type)
            curr_results["stump_is_right"]   = compare_answers(curr_results["stump_ans_format"],
                                                                curr_results["formatted_answer"],
                                                                self.answer_type)
            curr_results["stump_is_consistent"]   = compare_answers(curr_results["stump_ans_format"],
                                                                curr_results["orig_ans_format"],
                                                                self.answer_type)
            curr_results["is_consistent"]   = compare_answers(curr_results["no_deno_ans_format"],
                                                                curr_results["orig_ans_format"],
                                                                self.answer_type)
            curr_results["is_right"]        = compare_answers(curr_results["no_deno_ans_format"],
                                                                curr_results["formatted_answer"],
                                                                self.answer_type)
            curr_results["is_stump"]        = compare_answers(curr_results["no_deno_ans_format"],
                                                                curr_results["stump_ans_format"],
                                                                self.answer_type)
        
            self.results.append(curr_results.copy())
        return self.results
