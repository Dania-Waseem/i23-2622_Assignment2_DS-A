# CS-4063: Natural Language Processing — Assignment 2  
## Neural NLP Pipeline (From Scratch in PyTorch)

Student ID: i23-XXXX  
Course: CS-4063 NLP  
Institution: FAST NUCES  

---

## 📌 Overview
This project implements a full NLP pipeline from scratch in PyTorch:

- TF-IDF + PPMI representations  
- Skip-gram Word2Vec  
- BiLSTM + CRF for POS & NER  
- Transformer encoder for classification  

No pretrained models used.

---

## 📁 Repository Structure

i23-XXXX-NLP-Assignment2/
│
│── i23-XXXX_Assignment2_DS-X.ipynb
│── report.pdf
│
│── embeddings/
│   │── tfidf_matrix.npy
│   │── ppmi_matrix.npy
│   │── embeddings_w2v.npy
│   └── word2idx.json
│
│── models/
│   │── bilstm_pos.pt
│   │── bilstm_ner.pt
│   └── transformer_cls.pt
│
│── data/
│   │── pos_train.conll
│   │── pos_val.conll
│   │── pos_test.conll
│   │
│   │── ner_train.conll
│   │── ner_val.conll
│   └── ner_test.conll
│
└── README.md

---

## ⚙️ Installation

pip install torch numpy matplotlib scikit-learn tqdm

---

## 🚀 How to Run

### Part 1 — Word Embeddings
- TF-IDF matrix (10K vocab)
- PPMI matrix (window = 5)
- Skip-gram Word2Vec (K=10)
- t-SNE visualization
- Cosine similarity + analogy tests

Saved outputs:
embeddings/

---

### Part 2 — Sequence Labeling

Dataset:
- 500 annotated sentences
- POS: 12 tags
- NER: BIO scheme
- Stratified split:

  pos_train.conll  
  pos_val.conll  
  pos_test.conll  

  ner_train.conll  
  ner_val.conll  
  ner_test.conll  

Model:
- 2-layer BiLSTM
- CRF for NER
- Frozen vs fine-tuned embeddings
- Early stopping on val F1

---

### Part 3 — Transformer Encoder

- Scaled dot-product attention
- Multi-head attention (h=4)
- Sinusoidal positional encoding
- 4-layer Pre-LN encoder
- CLS classification head
- AdamW + cosine LR schedule

---

## 📊 Evaluation

- POS: Accuracy + Macro-F1 + Confusion Matrix  
- NER: Precision / Recall / F1 (entity-level)  
- Embeddings: MRR + nearest neighbors  
- Transformer: attention heatmaps  

---

## 🧪 Ablation Studies

A1 — Unidirectional LSTM  
A2 — No dropout  
A3 — Random embeddings  
A4 — Softmax instead of CRF  

---

## 📌 Notes

- Fully implemented in PyTorch from scratch  
- No pretrained models or libraries used  
- Padding masked properly  
- All training curves plotted  

---


