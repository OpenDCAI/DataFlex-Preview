---
library_name: peft
license: other
base_model: /data/malu/Qwen2.5-0.5B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: test
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# test

This model is a fine-tuned version of [/data/malu/Qwen2.5-0.5B-Instruct](https://huggingface.co//data/malu/Qwen2.5-0.5B-Instruct) on the alpaca_en_demo dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 16
- total_train_batch_size: 128
- total_eval_batch_size: 32
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1.0

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.50.0
- Pytorch 2.8.0+cu128
- Datasets 3.2.0
- Tokenizers 0.21.0