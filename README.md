# Examining feature localization in autoregressive language models

We provide a suite of tools to examine the phenomena of **feature localization**, where some aspect of the final generation appears during a narrow intervals of the sampling procedure of a stochastic localization sampler, for **autoregressive language models**. This codebase includes the code to reproduce the results in the manuscript [INSERT HERE]. 


## Installation
In order to install the required packages and dependences, use the following conda command.
```bash
conda env create -f environment.yml
```
We also use the `grade_answer` function from the `prm800k` library to grade the LLM math answers [CITATION].
```bash
git clone https://github.com/openai/prm800k.git
cd prm800k
pip install -e . 
```

## Structured output experiments
We describe our _structured output experiments_, where we explicitly compute the location of feature localization windows for a family of _simple outputs_. 

<img align="center" src="assets/structured_hierarchy.png" width="500">

The main challenge is to compute the total variation for different latents in a way that doesn't require too many samples. One way to do this is to use a  fill-in-the-blank style prompt. For example, here is the default prompt we provide in `configuration.py`. 
```
Complete the following by choosing only one option for each blank. The options are provided in parentheses, and your response must match the exact case and meaning of the chosen option. Respond with only the completed sentence, no explanations or additional text.
1. The (Pirate/Ninja) jumped across the ship.
2. She adopted a (Dog/Cat) from the shelter.
3. The (River/Bridge) sparkled under the sun.
4. A (Dragon/Knight) guarded the castle gates.
5. He ordered (Pizza/Sushi) for dinner.
```
An example language model response would be 
```
1. The Pirate jumped across the ship.
2. She adopted a Dog from the shelter.
3. The River sparkled under the sun.
4. A Dragon guarded the castle gates.
5. He ordered Pizza for dinner.
```
You can reproduce our experiments with the default prompt on Llama 3.1-8B using the following command.
```bash
## Structured output experiment
python experiments/structured_output/run_structured_output_experiments.py --model_id meta-llama/Llama-3.1-8B-Instruct --num_samples 10000
```
The results will be in `results/StructuredNoiseDenoise/StructuredNoiseDenoise_model=meta-llama-Llama-3.1-8B-Instruct_num_samples=10000`. The most important files in that folder are:
1. `structured_responses.csv` contains all the model responses and some useful information about the model responses
2. `ex_hierarchy/ex_{i}.png` are plots of the amount of truncation we apply versus the probability of samplign to the same piece of text. They also include the `Tlower` and `Tupper` bounds predicted by the theory.

## Chain of thought Reasoning experiments



## Jailbreak feature localization experiments
Here we describe our experiments to visualize different feature localization windows for different kinds of jailbreak. We consider both **prefill** jailbreaks and **repeat word** jailbreaks.

```bash
## Prefill jailbreak
python experiments/jailbreak/run_jailbreak_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset_type prefill_attack --num_per_noise 1
```

```bash
## Repeat word jailbreak
python experiments/jailbreak/run_jailbreak_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct  --dataset_type repeat_word --num_per_noise 100
```


## Jailbreak prompt detection method



