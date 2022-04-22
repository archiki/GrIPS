# GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models
* Authors: [Archiki Prasad](https://archiki.github.io), [Peter Hase](https://peterbhase.github.io/), [Xiang Zhou](https://owenzx.github.io/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/) (UNC Chapel Hill)
* [Paper](https://arxiv.org/abs/2203.07281)
* **Note:** This is preliminary version of our code. The complete code to run all experiments in the paper will be added shortly.

<img src="./assets/Main Pipeline.png" alt="teaser image" width="7500"/>

## Dependencies
This code is written using PyTorch and [HuggingFace's Transformer repo](https://github.com/huggingface/pytorch-transformers). Running GrIPS with GPT-2 models requires access to GPUs. The search is quite light-weight (no model training involved) and therefore one GPU should suffice. On the other hand, running GrIPS with InstructGPT or GPT-3 models requires an OpenAI API key. Please add your key to the `openai_key.txt` file.

## Installation
The simplest way to run our code is to start with a fresh environment.
```
conda create -n GrIPS python=3.9
source activate GrIPS
pip install -r requirements.txt
```

## Running Search
* `run_search.py` contains the implementation of GrIPS. 
  *  By default, we use the InstructGPT Babbage model. To use a different GPT-3 model from the API change `model_name` in `nat_inst_gpt3.py`.
  *  To switch to GPT-2 models, import `nat_inst_gpt2.py` and use an apporpriate model.
* `expanded_encodeinstructions.py` is a data loader file that interfaces with the task datasets provided in Natural Instructions.
* Here is an example code to run GrIPS (with default InstructGPT babbage)
```
python run_search.py --mode "Instruction Only" --task-idx 0 --train-seed 0 \
--num-compose 1 --num-candidates 5 --num-iter 10 --patience 2 --write-preds \
--meta-dir "logs/" --meta-name "babbage_all_edits_l_1_m_5_n_10@seed_0.txt"
```

## Acknowledgments
We thank the authors and contributors of [Callibrate Before Use](https://github.com/tonyzhaozh/few-shot-learning), and [Natural-Instructions](https://github.com/allenai/natural-instructions) for their public code release. 

## Reference
Please cite our paper if you use our dataset in your works:
```bibtex

@article{Prasad2022GrIPS,
  title         = {GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models},
  author        = {Archiki Prasad and Peter Hase and Xiang Zhou and Mohit Bansal},
  year          = {2022},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  eprint        = {2203.07281}
}
```
