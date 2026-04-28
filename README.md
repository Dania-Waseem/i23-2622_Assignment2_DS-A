# Urdu NLP Pipeline — End-to-End Text Understanding

This repository contains a complete end-to-end Natural Language Processing pipeline for Urdu news text.
The project covers dataset preparation, statistical representations, neural embeddings, sequence labeling, and transformer-based document classification — all implemented from scratch in PyTorch.

---

# Project Capabilities

This pipeline performs:

• Urdu text normalization and preprocessing
• Custom tokenization, stemming, and lemmatization
• TF-IDF document representations
• PPMI word co-occurrence embeddings
• Skip-gram Word2Vec training
• POS tagging using BiLSTM
• Named Entity Recognition using BiLSTM-CRF
• Transformer encoder for topic classification
• Attention visualization and embedding evaluation

All components are implemented without pretrained models.

---

# Repository Structure

```
urdu-nlp-pipeline/
│
├── models/
│   ├── bilstm_pos.pt
│   ├── bilstm_ner.pt
│   ├── bilstm_abl_a1.pt
│   ├── bilstm_abl_a2.pt
│   ├── bilstm_abl_a3.pt
│   └── transformer_cls.pt
│
├── files/
│   ├── Metadata.json
│   ├── raw.txt
│   ├── cleaned.txt
│   └── lemmatized.txt
│
├── embeddings/
│   ├── tfidf_matrix.npy
│   ├── ppmi_matrix.npy
│   ├── embeddings_w2v.npy
│   └── word2idx.json
│
├── data/
│   ├── pos_train.conll
│   ├── pos_val.conll
│   ├── pos_test.conll
│   ├── ner_train.conll
│   ├── ner_val.conll
│   └── ner_test.conll
│
├── urdu_language_modeling.ipynb
└── neural_urdu_nlp_pipeline.ipynb
```

---

# Pipeline Overview

The system is divided into three stages:

## 1. Text Processing

The dataset is normalized using:

• Unicode normalization
• Diacritics removal
• Noise filtering
• Non-Urdu removal
• Sentence segmentation
• Whitespace normalization

Custom linguistic tools:

• Urdu tokenizer
• Urdu stemmer
• Urdu lemmatizer

Outputs:

• cleaned.txt
• lemmatized.txt
• vocabulary mapping

---

# 2. Word Representation Learning

Three embedding methods are implemented:

## TF-IDF

Creates sparse document vectors using:

TF-IDF(w,d) = TF(w,d) × log(N / (1 + df(w)))

Used for:

• topic discriminative words
• baseline embeddings
• document similarity

Output:

tfidf_matrix.npy

---

## PPMI Embeddings

Word co-occurrence based representation:

PPMI(w1,w2) = max(0, log2( P(w1,w2) / (P(w1)P(w2)) ))

Used for:

• semantic similarity
• nearest neighbor search
• t-SNE visualization

Output:

ppmi_matrix.npy

---

## Skip-gram Word2Vec

Neural embedding model with:

• separate center and context matrices
• negative sampling
• binary cross entropy loss
• cosine similarity evaluation

Loss:

L = −logσ(uᵀv) − Σ logσ(−uₙᵀv)

Output:

embeddings_w2v.npy

---

# 3. Sequence Labeling

Two token-level tasks:

## POS Tagging

Tags:

NOUN, VERB, ADJ, ADV, PRON, DET, CONJ, POST, NUM, PUNC, UNK

Model:

Embedding → BiLSTM → Linear → Softmax

Evaluation:

• token accuracy
• macro F1
• confusion matrix

Model file:

bilstm_pos.pt

---

## Named Entity Recognition

BIO labels:

B-PER, I-PER
B-LOC, I-LOC
B-ORG, I-ORG
B-MISC, I-MISC
O

Architecture:

Embedding → BiLSTM → CRF → Viterbi decoding

Features:

• bidirectional context
• structured decoding
• transition matrix learning

Model file:

bilstm_ner.pt

---

# Ablation Models

Additional variants for analysis:

• bilstm_abl_a1.pt — unidirectional LSTM
• bilstm_abl_a2.pt — no dropout
• bilstm_abl_a3.pt — random embeddings

---

# Transformer Encoder

Document-level classification model using:

• scaled dot-product attention
• multi-head self attention
• positional encoding
• feed forward network
• residual connections
• layer normalization
• CLS classification token

Architecture:

Embedding

* Positional Encoding
  → 4 Encoder Blocks
  → CLS Token
  → MLP Classifier

Model file:

transformer_cls.pt

---

# Training Details

Embeddings:

dimension = 100
window size = 5
negative samples = 10

BiLSTM:

layers = 2
bidirectional = True
dropout = 0.5

Transformer:

heads = 4
hidden size = 128
FFN size = 512
encoder layers = 4

Optimizers:

Adam
AdamW

---

# Evaluation

Embedding evaluation:

• cosine similarity neighbors
• analogy tests
• t-SNE visualization

POS evaluation:

• accuracy
• macro F1
• confusion matrix

NER evaluation:

• precision
• recall
• F1
• CRF comparison

Transformer evaluation:

• classification accuracy
• macro F1
• confusion matrix
• attention heatmaps

---

# Requirements

Python 3.9+

Libraries:

```
numpy
torch
matplotlib
scikit-learn
```

---

# Running

Run notebooks in order:

1. urdu_language_modeling.ipynb
2. neural_urdu_nlp_pipeline.ipynb

The notebooks will:

• preprocess dataset
• train embeddings
• train BiLSTM models
• train transformer
• save outputs to folders

---

# Outputs

Embeddings saved in:

embeddings/

Models saved in:

models/

Annotated datasets saved in:

data/

Processed text saved in:

files/

---

# Notes

All models are trained from scratch.
No pretrained embeddings are used.
No transformer libraries are used.
All attention mechanisms are manually implemented.

---

# Author

Dania Waseem
