# Project Report

Here is a small report about what I have found, what I have thought, and what I have implemented in this version.

## Dataset Background

At the beginning of my research, I came across the **GYAFC** dataset (Grammarly’s Yahoo Answers Formality Corpus), introduced in this paper: [https://arxiv.org/abs/1803.06535](https://arxiv.org/abs/1803.06535). GYAFC is a widely used dataset for converting informal text to formal text. It consists of pairs of informal and formal sentences from Yahoo Answers. The original dataset covers two domains: Entertainment & Music and Family & Relationships. Since the full GYAFC dataset is not directly accessible, I initially used a subset available on GitHub: [https://github.com/luofuli/DualRL/tree/master/data/GYAFC](https://github.com/luofuli/DualRL/tree/master/data/GYAFC). However, it is worth considering using the complete dataset to capture a wider range of examples.

I also considered additional datasets for more examples of formal and informal speech. I used the **Enron email** dataset from HuggingFace: [https://huggingface.co/datasets/LLM-PBE/enron-email](https://huggingface.co/datasets/LLM-PBE/enron-email) for formal language. For informal language, I used comments from **Reddit** (under posts), which often contain slang, abbreviations, and a casual tone. Also the Dataset from HuggingFace: [https://huggingface.co/datasets/SocialGrep/the-reddit-dataset-dataset](https://huggingface.co/datasets/SocialGrep/the-reddit-dataset-dataset)

. Additionally, I generated a small dataset using **OpenAI's GPT model** to create 100 examples each of formal and informal speech. 

In total, I worked with three main datasets:
1. GYAFC subset (with potential to upgrade to the full dataset)
2. Emails/Reddit combined dataset
3. OpenAI-generated dataset

All scripts are available in the main folder. However, for convenience, I also leave the data in the data folder, so if you don't have an OpenAI key, the results can still be reproduced.

## Model Background

For selecting models suitable for formality detection, I used the MTEB (Massive Text Embedding Benchmark) [paper](https://arxiv.org/abs/2210.07316). This benchmark helps evaluate and compare various embedding models in natural language processing tasks. More details can be found on the MTEB leaderboard: [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard). I chose the five best-performing models from the leaderboard, specifically those with fewer than 1 billion parameters to ensure a balance between high performance and computational efficiency. The top three models are:

- **dunzhang/stella_en_400M_v5**
- **Alibaba-NLP/gte-large-en-v1.5**
- **HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1**

also I took **distilbert-base-uncased** as a very base model

## Evaluation Metrics

To evaluate the models effectively, I selected several widely used metrics in natural language processing:

- **ROC-AUC (Area Under the Receiver Operating Characteristic Curve)**: This metric shows the model's ability to distinguish between formal and informal text by measuring the trade-off between the true positive rate and the false positive rate.
- **Accuracy**: This metric shows the overall correctness of the model's predictions. It is simple and effective when the dataset is balanced.
- **F1-score**: This score combines precision and recall into a single measure. It is especially useful for unbalanced datasets, as it gives a better sense of how well the model is performing on both classes.


These metrics are popular in NLP because they provide a clear and comprehensive view of model performance in classification tasks.


## Additional Ideas [ oh, I wish I had more time :) ]

How to enhance the project :

- **Use the Full GYAFC Dataset**  
  Instead of relying on a subset, using the complete GYAFC dataset could provide a richer variety of examples and improve the robustness of the model.

- **Data Augmentation**  
  Adding data augmentation techniques, such as paraphrasing, back-translation, or synonym replacement, can create more diverse training examples. This helps the model learn better patterns and reduces overfitting.

- **Contrastive Learning for Embeddings**  
  Further training embedding models with contrastive learning techniques can improve their ability to differentiate between formal and informal text. This method creates more distinct representations of different text styles.

- **Addressing Domain Gaps**  
  The current setup includes data from emails and Reddit, which come from very different domains. This large domain gap may hurt model performance. A careful selection or further domain adaptation might be needed to reduce these differences.

- **Try basic NLP models**
  It would be also interesting to try some basic NLP models like Naive Bayes, SVM, etc. based on bm25 or tf-idf embeddings. However, it is an old-fashioned way to do things. Thus, I decided not to implement it for this project.

- **Use the zero-shot prompting**
  I wanted to use the zero-shot prompting on some Generative model to classify the text as formal or informal. (OpenAI API or any generative model from huggingface)

## What went wrong

The basic thing are special models and datasets for the task of formality detection. I have found some spesial traind models on huggingface, but they are not in the MTEB leaderboard, and there were only few of them, so I decided to not use them, because of possible data leakage. (for example, https://huggingface.co/AdapterHub/xlm-roberta-base_formality_classify_gyafc_pfeiffer, https://huggingface.co/laniqo/fame_mt_mdeberta_formality_classifer). The datasets were the problem as well, I have only foud the Ukrain version of the GYAFC dataset (https://huggingface.co/datasets/ukr-detect/ukr-formality-dataset-translated-gyafc), so I decided to not use it as well, and prepare mine own datasets. The Gpt generated dataset is good, but I don't have enough time (and money as well) to generate large amount of data. The datasets of emails and Reddit are not really good for prepared for the task, because they contain a lot of noise and they have a domain gap. That is why it is not very fair to compare the results of the models on this dataset.

Also, there are some models that I wanted to try, but their implementation on huggingface wasn't typical. The format is not the same as the other models, so I also decided to leave them out, because the issue of time.

- **jinaai/jina-embeddings-v3**
- **jxm/cde-small-v1**

GPU and time of evaluation and fine-tuning were the problem as well.
