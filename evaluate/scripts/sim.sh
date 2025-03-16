# Choose one of the following models:
# [distilbertbase,
# dunzhang/stella_en_400M_v5, 
# jinaai/jina-embeddings-v3, 
# Alibaba-NLP/gte-large-en-v1.5, 
# jxm/cde-small-v1, 
# HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1]

# Choose one of the following datasets:
# [gyafc, wikipedia_reddit, openai]

MODEL_NAME="dunzhang/stella_en_400M_v5" 
DATA_PATH="data/gyafc.txt"  # Change to your data file
OUTPUT_DIR="results/similarity"  # Change to your output directory

mkdir -p $OUTPUT_DIR

python evaluate/run_similarity.py \
  --model_name $MODEL_NAME \
  --data_path $DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --max_length 128 \
  --batch_size 32