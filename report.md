# Project Report

Here is a small report about what I have found, what I have thought, and what I have implemented in this version.

## Dataset Background

At the beginning of my research, I came across the GYAFC dataset (Grammarly’s Yahoo Answers Formality Corpus), introduced in this paper: [https://arxiv.org/abs/1803.06535](https://arxiv.org/abs/1803.06535). GYAFC is a widely used dataset for converting informal text to formal text. It consists of pairs of informal and formal sentences from Yahoo Answers. The original dataset covers two domains: Entertainment & Music and Family & Relationships. Since the full GYAFC dataset is not directly accessible, I initially used a subset available on GitHub: [https://github.com/luofuli/DualRL/tree/master/data/GYAFC](https://github.com/luofuli/DualRL/tree/master/data/GYAFC). However, it is worth considering using the complete dataset to capture a wider range of examples.

I also considered additional datasets for more examples of formal and informal speech. For example, I used the Enron email dataset from HuggingFace: [https://huggingface.co/datasets/LLM-PBE/enron-email](https://huggingface.co/datasets/LLM-PBE/enron-email) for formal language. For informal language, I used posts from Reddit, which often contain slang, abbreviations, and a casual tone. Additionally, I generated a small dataset using OpenAI's GPT model to create 100 examples each of formal and informal speech. In total, I worked with three main datasets:
1. GYAFC subset (with potential to upgrade to the full dataset)
2. Wikipedia/Reddit combined dataset
3. OpenAI-generated dataset

## Model Background

For selecting models suitable for formality detection, I used the MTEB (Massive Text Embedding Benchmark) [paper](https://arxiv.org/abs/2210.07316). This benchmark helps evaluate and compare various embedding models in natural language processing tasks. More details can be found on the MTEB leaderboard: [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard). I chose the five best-performing models from the leaderboard, specifically those with fewer than 1 billion parameters to ensure a balance between high performance and computational efficiency. The top five models are:

- **dunzhang/stella_en_400M_v5**
- **jinaai/jina-embeddings-v3**
- **Alibaba-NLP/gte-large-en-v1.5**
- **jxm/cde-small-v1**
- **HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1**

In addition to these models, I also evaluated performance using zero-shot prompting (https://huggingface.co/tasks/zero-shot-classification). This method uses pre-trained language models to classify text without further task-specific training, offering another way to measure how well a model understands formality. For this approach, the **OpenAI API** is used.

## Evaluation Metrics

To evaluate the models effectively, I selected several widely used metrics in natural language processing:

- **Accuracy**: This metric shows the overall correctness of the model's predictions. It is simple and effective when the dataset is balanced.
- **F1-score**: This score combines precision and recall into a single measure. It is especially useful for unbalanced datasets, as it gives a better sense of how well the model is performing on both classes.
- **ROC-AUC (Area Under the Receiver Operating Characteristic Curve)**: This metric shows the model's ability to distinguish between formal and informal text by measuring the trade-off between the true positive rate and the false positive rate.

These metrics are popular in NLP because they provide a clear and comprehensive view of model performance in classification tasks.

## Fine-Tuning

Fine-tuning pre-trained language models on the specific task of formality detection can further improve performance. For example, I experimented with fine-tuning **DistilBERT** on our dataset. However, this approach is not limited to DistilBERT; any pre-trained model can be fine-tuned for this task.

## Additional Ideas

Based on my extensive research, here are some data-focused ideas to further enhance the project:

- **Use the Full GYAFC Dataset**  
  Instead of relying on a subset, using the complete GYAFC dataset could provide a richer variety of examples and improve the robustness of the model.

- **Data Augmentation**  
  Adding data augmentation techniques, such as paraphrasing, back-translation, or synonym replacement, can create more diverse training examples. This helps the model learn better patterns and reduces overfitting.

- **Contrastive Learning for Embeddings**  
  Further training embedding models with contrastive learning techniques can improve their ability to differentiate between formal and informal text. This method creates more distinct representations of different text styles.

- **Addressing Domain Gaps**  
  The current setup includes data from emails and Reddit, which come from very different domains. This large domain gap may hurt model performance. A careful selection or further domain adaptation might be needed to reduce these differences.

