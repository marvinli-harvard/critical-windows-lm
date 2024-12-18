# Examining Feature Localization in autoregressive language models

In this repo, we provide a suite tools to examine the phenomena of feature localization, where some aspect of the final generation, emerges during a narrow intervals of the sampling procedure. In particular, we focus on the setting of langauge models. This codebase includes the experiments to reproduce the results in the manuscript [INSERTHERE]. 


## Installation
In order to install the required packages and dependences, use the following conda command.
```bash
conda env create -f environment.yml
```
We also use the `grade_answer` function from the `prm800k` library [CITATION]. This can be 
```bash
git clone https://github.com/openai/prm800k.git
cd prm800k
pip install -e . 
```

## Structured output experiments
In this section, we will describe the experiments to identify the location of feature localization for a class of simplified outputs. The challenge of these experiments is computing the total variation for two different latents at a given level of noising and denoising in a way that doesn't require too many samples. Here, we use a prompt that restricts the number of different outputs of the language model. Below is the default prompt that we provide in `configuration.py`. 
> Complete the following by choosing only one option for each blank. The options are provided in parentheses, and your response must match the exact case and meaning of the chosen option. Respond with only the completed sentence, no explanations or additional text.
> 1. The (Pirate/Ninja) jumped across the ship.
> 2. She adopted a (Dog/Cat) from the shelter.
> 3. The (River/Bridge) sparkled under the sun.
> 4. A (Dragon/Knight) guarded the castle gates.
> 5. He ordered (Pizza/Sushi) for dinner.

You can reproduce our experiments with this prompt on Llama 3.1-8B using the following command.
```bash
python experiments/structured_output/run_structured_output_experiments.py --model_id meta-llama/Llama-3.1-8B-Instruct --num_samples 10000
```
The script will output the experimental results in `results/StructuredNoiseDenoise/StructuredNoiseDenoise_model=meta-llama-Llama-3.1-8B-Instruct_num_samples=10000`. The most important files in that folder are as follows:
1. `structured_responses.csv` contains all the model responses and some useful information about the model responses
2. `ex_hierarchy/ex_{i}.png` are the images of the probability of a yielding the same text as a function of the amount of truncation we apply to a given piece of text. They also plot the `Tlower` and `Tupper` bounds predicted by the theory.


## Chain of thought reasoning experiments



## Prefill jailbreak feature localization experiments



## Jailbreak prompt detection method



