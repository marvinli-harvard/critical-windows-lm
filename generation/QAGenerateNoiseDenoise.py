from typing import Dict, List, Optional, Tuple

import transformers

from datasets import Dataset, load_dataset
from vllm import SamplingParams

from utils.configuration import *
from utils.utils import *
from utils.dataset_utils import *
from utils.generation_utils import *
from utils.grader_utils import *

from prompt_generation.PromptGenerationBase import PromptGenerationBase
from generation.GenerationBase import GenerationBase

class QAGenerateNoiseDenoise(GenerationBase):
    def __init__(self, dataset : Dataset, answer_type  : str, **kwargs):
        self.dataset = dataset 
        self.answer_type = answer_type
        super().__init__(**kwargs)

    def generate_basic(self) -> List[Dict]:
        first_responses = []
        question_tokens = [self.prompt_gen.get_question_tokens(item) for item in self.dataset["problem"]]
        outputs, responses = self.generate(question_tokens, self.genwrapper.sampling_first)
        for i, val in tqdm(list(enumerate(self.dataset.iter(batch_size=1)))):
            curr_value = {k:v[0] for k,v in val.items()}
            asst_index = torch.where(outputs[i][0]==self.prompt_gen.heading_to_tokens["assistant"])[0][0]+2
            
            curr_value["question_tokens"] = question_tokens[i]
            curr_value["orig_tokens"] = outputs[i][0]
            curr_value["orig_string"] = responses[i][0]
            curr_value["asst_tokens"] = outputs[i][0][asst_index:]
            first_responses.append(curr_value.copy())
        return first_responses
    
    def generate_noised_denoised_questions(self, first_responses: List[Dict],
                                           percent_prompt : List[float]) -> List[Dict]:
        noised_questions_tokens = []
        for gens in tqdm(first_responses):
            for stop_frac in percent_prompt:
                entry = gens.copy()
                entry["stop_frac"] = stop_frac
                inputs = self.prompt_gen.get_noise_denoise_question(question=entry["problem"], 
                                                            response_tokens=entry["asst_tokens"].tolist(),
                                                            stop_frac=stop_frac)
                entry["no_deno_input_tokens"] = list(inputs)
                entry["no_deno_input_string"] = self.genwrapper.tokenizer.decode(inputs, skip_special_tokens=False)
                noised_questions_tokens.append(entry)
        return noised_questions_tokens
    
    def generate_noised_denoised(self, first_responses : List[Dict],
                                 percent_prompt : List[float]) -> List[Dict]:
        noised_questions_tokens = self.generate_noised_denoised_questions(first_responses=first_responses,
                                                                          percent_prompt=percent_prompt)
        
        ## Now run generation
        question_tokens = [item["no_deno_input_tokens"] for item in noised_questions_tokens]
        outputs, responses = self.generate(question_tokens, self.genwrapper.sampling_repeat)
        noised_denoised_results = []
        for i in tqdm(range(len(outputs))):    
            batch_output   = outputs[i]
            batch_response = responses[i]
            for j in range(len(batch_output)):
                curr_value = noised_questions_tokens[i].copy()
                curr_value["no_deno_output_tokens"]     = batch_output[j]
                curr_value["no_deno_output_strings"]    = batch_response[j]
                curr_value["no"]                        = j
                noised_denoised_results.append(curr_value)
        return noised_denoised_results
    
    def generate_orig_stump(self, noised_denoised_results : List[Dict])->Tuple[Dict, Dict]:
        existing_orig_responses  = {}
        existing_stump_responses = {}
        for example in noised_denoised_results:
            if example["problem"] not in existing_orig_responses:
                existing_orig_responses[example["problem"]] = \
                    self.prompt_gen.complete_with_answer(example["orig_tokens"].tolist())
            if tuple(example["no_deno_input_tokens"]) not in existing_stump_responses:
                existing_stump_responses[tuple(example["no_deno_input_tokens"])] = \
                    self.prompt_gen.complete_with_answer(example["no_deno_input_tokens"] + [self.prompt_gen.heading_to_tokens["eot_id"]])
        return existing_orig_responses, existing_stump_responses

    def format_answer(self, prefix: str, token, output, response) -> Dict:
        """Formats an answer with the given prefix."""
        new_tokens = output[len(token):-1]
        new_string = self.genwrapper.tokenizer.decode(new_tokens)
        return {
            f"{prefix}_tokens_ans": output,
            f"{prefix}_string_ans": response,
            f"{prefix}_ans_tokens": new_tokens,
            f"{prefix}_ans_string": new_string,
            f"{prefix}_ans_format": self.extract_answer(new_string),
        }
    
    def generate_answers(self, inputs: Dict, prefix: str) -> Dict[str, Dict]:
        """Generates answers based on the given inputs and a prefix."""
        answers = {}
        tokens = [item[1] for item in inputs.items()]
        outputs, model_responses = self.generate(tokens, self.genwrapper.sampling_answer)

        for key, token, output, response in zip(inputs.keys(), tokens, outputs, model_responses):
            answers[key] = self.format_answer(prefix, token, output[0], response[0])

        return answers
    def generate_original_answers(self, existing_orig_responses: Dict) -> Dict[str, Dict]:
        """Generates original answers."""
        return self.generate_answers(existing_orig_responses, "orig")

    def generate_stump_answers(self, existing_stump_responses: Dict) -> Dict[str, Dict]:
        """Generates stump answers."""
        return self.generate_answers(existing_stump_responses, "stump")
    
    def generate_noised_denoised_answers(self, 
                                         noised_denoised_results: List[Dict],
                                         orig_answers: Dict,
                                         stump_answers: Dict) -> List[Dict]:
        """Generates answers for noised and denoised inputs."""
        results = []
        noised_denoised_tokens = [
            self.prompt_gen.complete_with_answer(b["no_deno_output_tokens"].tolist())
            for b in noised_denoised_results
        ]

        outputs, responses = self.generate(noised_denoised_tokens, self.genwrapper.sampling_answer)

        for result, output, response, tokens in zip(noised_denoised_results, outputs, responses, noised_denoised_tokens):
            curr_results = result.copy()
            curr_results.update(self.format_answer("no_deno", tokens, output[0], response[0]))

            problem_key = curr_results.get("problem", None)
            input_tokens_key = tuple(curr_results.get("no_deno_input_tokens", []))
            
            curr_results.update(orig_answers.get(problem_key, {}))
            curr_results.update(stump_answers.get(input_tokens_key, {}))

            curr_results.update(self.evaluate_consistency_and_accuracy(curr_results))
            results.append(curr_results)

        return results
    
    def evaluate_consistency_and_accuracy(self, curr_results: Dict) -> Dict:
        """Evaluates the consistency and correctness of the answers."""
        return {
            "orig_is_right": self.compare_answers(
                curr_results.get("orig_ans_format"),
                curr_results.get("formatted_answer")
            ),
            "stump_is_right": self.compare_answers(
                curr_results.get("stump_ans_format"),
                curr_results.get("formatted_answer")
            ),
            "stump_is_consistent": self.compare_answers(
                curr_results.get("stump_ans_format"),
                curr_results.get("orig_ans_format")
            ),
            "is_consistent": self.compare_answers(
                curr_results.get("no_deno_ans_format"),
                curr_results.get("orig_ans_format")
            ),
            "is_right": self.compare_answers(
                curr_results.get("no_deno_ans_format"),
                curr_results.get("formatted_answer")
            ),
            "is_stump": self.compare_answers(
                curr_results.get("no_deno_ans_format"),
                curr_results.get("stump_ans_format")
            ),
        }
    
    def extract_answer(self, answer_string: str):
        """Extracts the formatted answer."""
        return extract_answer(answer_string, self.answer_type)

    def compare_answers(self, answer1, answer2):
        """Compares two answers for consistency or correctness."""
        return compare_answers(answer1, answer2, self.answer_type)
