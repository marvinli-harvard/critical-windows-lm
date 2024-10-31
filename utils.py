import numpy as np
import torch
import transformers
from datasets import load_dataset
from typing import Dict
import os
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ACCESS_TOKEN = os.environ["HF_ACCESS_TOKEN"]
MATH_DATASET="/content/MATH"
ANSWER_LOCATOR = "###"
SYSTEM_PROMPT = \
    f"You answer math problems. After you finish thinking through the math problem, \
    place your answer between two `{ANSWER_LOCATOR}`. For example, if the answer is 26, then your \
    response should end with {ANSWER_LOCATOR}26."

def load_model_pipeline(model_id):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
        token=ACCESS_TOKEN
    )
    return pipeline

def extract_between(before, after,text):
    return text.split(before)[-1].split(after)[0]

def text_to_response(sample,
                     pipeline,
                     system_prompt=SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=4096
    )
    return outputs[0]

def question_to_response(sample, pipeline,
                         system_prompt=SYSTEM_PROMPT):
    return text_to_response(sample["problem"], pipeline, system_prompt)

def extract_answer_from_response(response):
    return response.split(ANSWER_LOCATOR)[-2] if len(response.split(ANSWER_LOCATOR)) > 2 else None

def get_raw_tokens_from_response(text :str , tokenizer):
    return tokenizer(text, add_special_tokens=False)["input_ids"]

heading_to_tokens = {
        "system":9125,
        "user":882,
        "assistant":78191
    }
eot_id = [128009]

def return_heading_list_llama(type :str, start=False):
    if start:
        return [
                128000,  # <|begin_of_text|>
                128006,  # <|start_header_id|>
                heading_to_tokens[type],
                128007,  # <|end_header_id|>
                271,     # "\n\n"
        ]
    else:
        return [
                128006,  # <|start_header_id|>
                heading_to_tokens[type],
                128007,  # <|end_header_id|>
                271,     # "\n\n"
        ]

class ResponseGenerator:
    def __init__(self, response, tokenizer, system_prompt:str):
        self.response       = response
        self.answer         = extract_answer_from_response(response["generated_text"][-1]["content"])
        self.raw_tokens     = get_raw_tokens_from_response(response["generated_text"][-1]["content"], tokenizer)
        self.length         = len(self.raw_tokens)
        self.tokenizer      = tokenizer
        self.system_prompt  = system_prompt

    def get_message(self, question:str, stop_frac:float=0.2):
        system_heading           = return_heading_list_llama("system",
                                                             start=True)
        system_tokens            = get_raw_tokens_from_response(self.system_prompt,
                                                                self.tokenizer)
        user_heading             = return_heading_list_llama("user",
                                                             start=True)
        user_tokens              = get_raw_tokens_from_response(question,
                                                                self.tokenizer)
        assistant_heading        = return_heading_list_llama("assistant",
                                                             start=True)
        assistant_tokens         = self.get_after_percent(stop_frac=stop_frac)

        return system_heading + system_tokens + eot_id  + \
                user_heading + user_tokens + eot_id + \
                assistant_heading + assistant_tokens


    def get_after_percent(self, stop_frac:float=0.2):
        return self.raw_tokens[:int(stop_frac * self.length)]

def get_average(responseGenerator, problem, stop_frac, pipeline, bs=16):
    message_tokenized = responseGenerator.get_message(problem,stop_frac=stop_frac)
    model_generation = pipeline.model.generate(
        torch.tensor([message_tokenized for _ in range(bs)]).to(device),
        max_length=4096)
    decoded = pipeline.tokenizer.batch_decode(model_generation.cpu().numpy())
    return [extract_between(ANSWER_LOCATOR,"<|eot_id|>", dec) for dec in decoded]

def get_noised_samples(math_question, pipeline, stop_fracs, system_prompt = SYSTEM_PROMPT, nbs = 10, bs=64):
    from tqdm import tqdm

    response = question_to_response(sample=math_question,pipeline=pipeline)
    raw_tokens = get_raw_tokens_from_response(response["generated_text"][-1]["content"],
                                          pipeline.tokenizer)
    responseGenerator = ResponseGenerator(response, pipeline.tokenizer, system_prompt)
    dict_responses = {"orig":response}
    for stop_frac in tqdm(stop_fracs):
        dict_responses[stop_frac] = []
        for _ in range(nbs):
            dict_responses[stop_frac].extend(get_average(responseGenerator, math_question["problem"], stop_frac, pipeline, bs = bs))
    return dict_responses


