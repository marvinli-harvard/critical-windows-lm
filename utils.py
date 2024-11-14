import numpy as np
import torch
import transformers
from datasets import load_dataset
from typing import Dict
import os
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ACCESS_TOKEN = os.environ["HF_ACCESS_TOKEN"]
SYSTEM_PROMPT = "Produce a correct solution to the following /TASK/ question."
COT_PROMPT = \
    "Think of the /TASK/ question thoroughly step by step. \
    Please only respond with the answer after reasoning thoroughly. \n\n"

def load_model_pipeline(model_id, 
                        generator=None,
                        temperature=0.6):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
        temperature=temperature,
        token=ACCESS_TOKEN,
        low_cpu_mem_usage=False,
        generator=generator
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


def get_raw_tokens_from_response(text :str , tokenizer):
    return tokenizer(text, add_special_tokens=False)["input_ids"]

# def extract_answer_from_response(response):
#     return response.split(ANSWER_LOCATOR)[-2] if len(response.split(ANSWER_LOCATOR)) > 2 else None

# def get_average(responseGenerator, problem, stop_frac, pipeline, bs=16):
#     message_tokenized = responseGenerator.get_message(problem,stop_frac=stop_frac)
#     model_generation = pipeline.model.generate(
#         torch.tensor([message_tokenized for _ in range(bs)]).to(device),
#         max_length=4096)
#     decoded = pipeline.tokenizer.batch_decode(model_generation.cpu().numpy())
#     return [extract_between(ANSWER_LOCATOR,"<|eot_id|>", dec) for dec in decoded]

# def get_noised_samples(math_question, pipeline, stop_fracs, system_prompt = SYSTEM_PROMPT, nbs = 10, bs=64):
#     from tqdm import tqdm

#     response = question_to_response(sample=math_question,pipeline=pipeline)
#     raw_tokens = get_raw_tokens_from_response(response["generated_text"][-1]["content"],
#                                           pipeline.tokenizer)
#     responseGenerator = ResponseGenerator(response, pipeline.tokenizer, system_prompt)
#     dict_responses = {"orig":response}
#     for stop_frac in tqdm(stop_fracs):
#         dict_responses[stop_frac] = []
#         for _ in range(nbs):
#             dict_responses[stop_frac].extend(get_average(responseGenerator, math_question["problem"], stop_frac, pipeline, bs = bs))
#     return dict_responses


