# bandit-qa
Code for [_Simulating Bandit Learning from User Feedback for Extractive Question Answering_](https://arxiv.org/pdf/2203.10079.pdf). 

Please contact the first author if you have any questions.

## Table of Contents
- [Basics](#basics)
- [Data](#data)
- [Installation](#installation)
- [Instruction](#instruction)
- [Citation](#citation)

## Basics
Brief intro for each file:
- train.py: training script 
- model.py: model implementaton
- util*.py: codes for evaluation


## Data
You can download MRQA datasets from [MRQA official repo](https://github.com/mrqa/MRQA-Shared-Task-2019#training-data): training data and in-domain development data. 

You can download small sets of supervised examples for initial training from [Splinter repo](https://github.com/oriram/splinter): follow the instruction under "Downloading Few-Shot MRQA Splits".

We suggest you to create a _data_ folder and save all data files there. 


## Installation
1. This project is developed in Python 3.9.5. Using Conda to set up a virtual environment is recommended.

2. Install the required dependencies. 
    ```
    pip install -r requirements.txt
    ```
3. Install PyTorch from http://pytorch.org/.


## Instruction
You can run the following command to start an online simulation experiment with wandb logging:

```
python train.py --notes 'your own notes for this experiment if needed' --wandb --do_train --do_eval --model SpanBERT/spanbert-base-cased --seed 46 --train_file ??? --dev_file ??? --max_seq_length 512 --doc_stride 128 --eval_metric f1 --num_train_epochs 1 --eval_per_epoch 8 --output_dir .simulation --initialize_model_from_checkpoint ??? --train_batch_size 80 --eval_batch_size 20 --gradient_accumulation_steps 4 --scheduler constant --algo 'R' --turn_off_dropout --argmax_simulation
```


??? means the path to file needed by the argument. Please read the argparse code at the bottom of train.py to understand what arguments you could further configure. 

To obtain a model initially trained on some supervised data, you are welcome to 1) use training scripts in [SpanBERT repo](https://github.com/facebookresearch/SpanBERT) on SQuAD and MRQA datasets, or 2) configure train.py in this repo.

## Citation
```
@inproceedings{Gao2022:banditqa-simulation,
  title     = {Simulating Bandit Learning from User Feedback for Extractive Question Answering},
  author    = {Gao, Ge  and
               Choi, Eunsol  and
               Artzi, Yoav},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  month     = may,
  year      = {2022},
  address   = {Dublin, Ireland},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2022.acl-long.355},
  pages     = {5167--5179}
}
```
