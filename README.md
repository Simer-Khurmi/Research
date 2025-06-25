# 🧠 HTHTA: A Hybrid Temporal–Hierarchical Transformer-Attention Framework for Cross-Domain Time-Sensitive Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Transformer](https://img.shields.io/badge/Architecture-Transformer--Hybrid-purple)
![Status](https://img.shields.io/badge/Status-Under%20Review-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **HTHTA is the first unified deep learning framework that generalizes across domains—solar forecasting, text classification, and clinical prediction—without redesigning architecture per task.**

---

## 🔍 Abstract

HTHTA integrates three core components:

- 🌲 **Random Forest-based temporal embedding** for structured tabular time-series
- 🤖 **BERT encoder** for text semantics
- 🧠 **BiLSTM + Hierarchical Attention** for universal downstream prediction

It achieves:
- 📉 RMSE `0.00048 kW` in solar forecasting (29% better than CNN-LSTM)
- 📰 94.6% accuracy on AG News (0.3% > RoBERTa)
- 🩺 AUC `0.87` for heart failure prognosis (XGBoost = 0.85)

---

## ⚙️ Methodology

```mermaid
graph TD
    A1[Structured Data] --> B1[Random Forest Embedding]
    A2[Text Data] --> B2[BERT Encoder]
    B1 --> C[BiLSTM Layer]
    B2 --> C
    C --> D[Hierarchical Attention]
    D --> E[Prediction Head (Regression or Classification)]
