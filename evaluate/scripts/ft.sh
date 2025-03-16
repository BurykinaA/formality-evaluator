#!/bin/bash

# Choose one of the following models:
# [distilbert-base-uncased,
# dunzhang/stella_en_400M_v5, +
# Alibaba-NLP/gte-large-en-v1.5, +
# HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1] +

# Choose one of the following datasets:
# [gyafc, reddit_enron_combined, gpt_generated]

MODEL_NAME="distilbert-base-uncased"  
DATA_DIR="data/gpt_generated.txt"   
OUTPUT_DIR="results/finetune"  

mkdir -p $OUTPUT_DIR

python evaluate/run_finetune.py \
  --model_name $MODEL_NAME \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --num_epochs 1 \
  --max_length 128 \
  --pooling cls 

#  --save_model

