# nutriBERT

A machine learning project for predicting carbohydrate content in meals from natural language descriptions using multiple approaches including TF-IDF, Sentence-BERT, and fine-tuned BERT models.

## Overview

This project explores three different machine learning approaches to predict carbohydrate content from meal descriptions:

1. **TF-IDF + Ridge Regression** - Traditional NLP approach using term frequency-inverse document frequency vectorization
2. **Sentence-BERT + MLP** - Using pre-trained sentence embeddings with a multi-layer perceptron
3. **Fine-tuned BERT Regressor** - End-to-end BERT model with a custom regression head

## Goal

Predict the carbohydrate content of a meal based on its natural language description using machine learning models trained on a labeled dataset.

## Model Performance

| Model | Validation MAE |
|-------|----------------|
| TF-IDF + Ridge Regression | 14.77 |
| Sentence-BERT + MLP | 9.87 |
| Fine-tuned BERT (5 epochs) | 10.46 |
| Fine-tuned BERT (10 epochs) | 8.73 |
| Fine-tuned BERT (20 epochs) | **7.06** |

The fine-tuned BERT model achieved the best performance with a validation MAE of 7.06 after 20 epochs of training.

```

### Key Libraries

- **transformers**: For BERT model implementation
- **sentence-transformers**: For pre-trained sentence embeddings
- **scikit-learn**: For traditional ML models and evaluation
- **torch**: For deep learning framework
- **pandas**: For data manipulation
- **matplotlib**: For visualization


## Training Process

### BERT Fine-tuning

The BERT model was trained with the following configuration:
- **Model**: `bert-base-uncased`
- **Optimizer**: AdamW with learning rate 2e-5
- **Loss Function**: L1 Loss (Mean Absolute Error)
- **Batch Size**: 16
- **Epochs**: 20
- **Max Length**: 64 tokens
