#!/bin/bash


MODEL_NAME="bert-base-uncased"  
DATA_DIR="data/gyafc.txt"   
OUTPUT_DIR="results/finetune"  

mkdir -p $OUTPUT_DIR

python evaluate/run_finetune.py \
  --model_name $MODEL_NAME \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --num_epochs 5 \
  --max_length 128 \
  --pooling cls \
  --save_model

