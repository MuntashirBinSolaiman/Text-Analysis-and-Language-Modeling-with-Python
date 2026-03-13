# Text Analysis and Language Modeling with Python

## What
This project explores key Natural Language Processing (NLP) tasks, including:
- Computing TF-IDF vectors to represent document importance.
- Classifying COVID-19 tweets into sentiment categories using word embeddings.
- Fine-tuning a DistilBERT transformer for part-of-speech (POS) tagging.

The project demonstrates both classical and modern NLP techniques using Python.

## How
The project was implemented using Python with popular libraries:
- **TF-IDF**: Term frequencies and inverse document frequencies were computed using Pandas and NumPy.
- **Sentiment Classification**: Tweets were preprocessed with NLTK, transformed into 100-dimensional GloVe embeddings, and classified using Logistic Regression and a PyTorch neural network.
- **POS Tagging**: A pre-trained DistilBERT model from Hugging Face was fine-tuned. Data was tokenized with alignment for subwords, and training was managed using Hugging Face's Trainer class.

Evaluation metrics included accuracy, precision, recall, and F1 scores.

## Why
The project illustrates the progression from traditional NLP techniques to modern transformer-based models:
- TF-IDF provides foundational insight into text representation.
- Word embeddings enable capturing semantic meaning for classification tasks.
- Fine-tuning transformers like DistilBERT demonstrates state-of-the-art NLP capabilities, producing high accuracy in syntactic tasks such as POS tagging.

This showcases how different approaches can be applied depending on the complexity and requirements of a text-based task.
