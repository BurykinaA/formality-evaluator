## Information about the project

For detailed information about the project, please refer to:
- [Project Report](report.md)
- [Documentation](documentation.md)

## Installation
### Using Poetry (Recommended)

1. Install Poetry if you don't have it already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/BurykinaA/formality-evaluator.git
   cd formal-informal-classification
   ```

3. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

4. Activate the Poetry environment:
   ```bash
   poetry shell
   ```


### Hardware Requirements

- **GPU Required**: This project requires a GPU for efficient model training and evaluation, especially when working with transformer models.

## Data Generation

Note: This step is optional, because the data is already generated.

1. Set your OpenAI API key in the environment variables:
   ```bash
   export OPENAI_API_KEY=your_api_key
   ```

2. Generate synthetic data:
   ```bash
   python generate_gpt_data.py
   ```

3. Process GYAFC data:
   ```bash
   python gyafc_generate.py
   ```

4. Prepare Reddit and Enron datasets:
   ```bash
   python prepare_datasets.py
   ```

## Running Experiments

### Fine-tuning Approach

```bash
evaluate/scripts/ft.sh
```

### Similarity-based Approach

```bash
evaluate/scripts/similarity.sh
```

## Results


### Metrics: ROC-AUC

Note: all results are located in the `results` folder.

### finetune

| Model                      | gpt_generated | gyafc | reddit_enron_combined (F1-score) |
|----------------------------|--------------|-------|------------------------|
| gte-large-en-v1.5          |       1.0       |   0.9131    | 0.9973                       |
| KaLM-embedding-multilingual |      1.0    |   0.7234    | 0.9975                       |
| stella_en_400M_v5          |      1.0      |   0.8787   | 0.9981                     |
| distilbert-base-uncased    | in progress  | 0.8864 |  in progress                      |

### similarity

| Model                      | gpt_generated | gyafc | reddit_enron_combined |
|----------------------------|--------------|-------|------------------------|
| gte-large-en-v1.5          |       0.5964       |   0.4742    | in progress                       |
| KaLM-embedding-multilingual |       0.1226       | 0.4690      | in progress                       | 
| stella_en_400M_v5          |      0.1737        |  0.4006    |   in progress                     |
| distilbert-base-uncased    |     0.1645         |  0.3721     |  in progress                     |

### Conclusion
From the experimental results, it is clear how important it is to train embedding models. I believe that if there were a zero-shot approach, the results would be better (when considering a method without fine-tuning the models).  

I am still in the process of scoring the results for the largest dataset (**reddit_enron_combined**).  

Regarding fine-tuning, it is evident that for the **gpt_generated** dataset, all models have overfitted (who would have thought? 🙂). For **gyafc**, the best result was achieved by **gte-large-en-v1.5**.