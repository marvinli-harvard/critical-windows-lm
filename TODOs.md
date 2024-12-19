### Coding modifications/re-running
- [ ] Plots about breakdown of correctness as a function of the length of the critical window
- [ ] Move COT and jailbreak plots from jupyter notebook to scripts
- [ ] Run all experiments on Gemini-7b instruct

### Improve rigor of jailbreak experiments
- [ ] Use dataset from Luke Bailey paper to compute jailbreak accuracy
- [ ] Test using pre-instruct model instead of the jailbroken experiments

### Misc. to-dos
- [ ] why narrow exposition
- [ ] Read safe token paper

## DONE

### Descriptive experiments
- [X] Overall percentages (reminder to reproduce because shuffle was not applied for all but one dataset)\
        - [X] Compare with just asking the model for the answers \
- [X] Run curves for 400 examples from each dataset\
        - [X] Look at specific examples and find some sort of pattern
- [X] Generate critical windows for jailbreaks 
- [X] Generate critical windows for structured data 

### Methods/prescription to try
- [x] Likelihood ratio between jailbroken and not jailbroken model to predict prob of jailbreak behavior
- [x] See if prompting LLM with critical windows makes it easier to correct its mistakes

## DONE
- [X] Implement AquA
- [X] Include correctness information
- [X] Batchify noise denoise
- [X] Make eval to compare with true answer & work with batch
    -   Check that new graders reduce number of Nones
- [X] Run on 10 individual samples with lots of data from each noising and denoising time level
- [X] Convert everything to vllm
- [X] Write code to compare with asking for answer directly
- [X] Run on 400 samples with lots of data from each noising and denoising time level
- [X] Make `is_stump` with `is_consistent` plot
- [X] Overall diagrams for 10k dataset
- [X] Refactor some code
- [X] Explore critical windows CoT and come up with some sort of explanation/hypothesis - Important parts of the reasoning process
- [X] Construct dataset with different jailbreaks and plot critical windows for jailbreaking\
- [X] structured data 
- [X] See if reminding LLM of critical windows makes it better able to correct
- [X] Likelihood ratio between jailbroken and not jailbroken model to predict prob of jailbreak behavior
- [X] Fix increased size of dataset from 1k to 1.008k during merge of `instruction` of aligned data for GPT
- [x] get rid of madlib wording because it isnt accurate 
- [x] plot frequency of jailbreaks/info regarding jumps 
- [x] Read phi, bailey, and safety token training papers

