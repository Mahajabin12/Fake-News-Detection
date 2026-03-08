# Fake News Detection using RoBERTa + Streamlit (with Explainable AI)

This repository contains an end-to-end **Fake News Detection** project built with:
- **RoBERTa-base** fine-tuned for binary classification (**FAKE = 0**, **REAL = 1**)
- A **Streamlit web application** with a 3-step flow (Intro → Predict → Result)
- **Explainable AI (XAI)** using **Integrated Gradients** to show how the model made a decision
- Model hosted on **Hugging Face Hub** to avoid GitHub’s **25MB file limit**
- Dataset included in this GitHub repository inside the project folder

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Label Mapping](#label-mapping)
4. [Models Compared](#models-compared)
5. [Final Model Choice](#final-model-choice)
6. [Training Pipeline (A to Z)](#training-pipeline-a-to-z)
7. [Why the Model is on Hugging Face](#why-the-model-is-on-hugging-face)
8. [Streamlit App (How it Works)](#streamlit-app-how-it-works)
9. [Explainable AI (Integrated Gradients)](#explainable-ai-integrated-gradients)
10. [Project Structure](#project-structure)
11. [Installation](#installation)
12. [Run the Streamlit App](#run-the-streamlit-app)
13. [How to Re-train the Model](#how-to-re-train-the-model)
14. [Upload / Update Model on Hugging Face](#upload--update-model-on-hugging-face)
15. [Troubleshooting](#troubleshooting)
16. [Disclaimer](#disclaimer)

---

## Project Overview
Fake news spreads quickly through social media and online platforms, making automatic detection important.  
This project uses a transformer-based NLP model (RoBERTa) to classify an input news text as:

- **FAKE (0)**: likely misinformation
- **REAL (1)**: likely factual / reliable

The system includes:
- Training + evaluation scripts (your notebook / training code)
- A Streamlit interface for real-time testing
- Explainable AI visualizations to make the model decisions easier to understand

---

## Dataset
The dataset used in this project is included in this repository and stored inside the project folder.

- **File:** `df_cleaned_verified.csv` (or your dataset filename)
- **Columns:**
  - `text` → news content
  - `marks` → class label

Your final dataset after cleaning:
- Rows: ~47,758
- Columns: 2 (`text`, `marks`)
- No missing values
- Very small number of duplicates removed

### Class balance (important)
Your dataset distribution is close to balanced:
- REAL (1): ~54%
- FAKE (0): ~46%

This is typically **good enough**, and you do not strictly need oversampling/undersampling.

---

## Label Mapping
This project uses the following label mapping everywhere:

- **FAKE = 0**
- **REAL = 1**

---

## Models Compared
You trained and evaluated three transformer models:

1. **DeBERTa-v3-base** (advanced / strong performance)
2. **RoBERTa-base** (reliable and very strong for news text)
3. **DistilBERT-base-uncased** (smaller + faster, slightly lower accuracy)

### Your results (summary)
From your outputs, all models achieved around **~0.97–0.98 accuracy**, but:

- **RoBERTa-base** gave the best overall test performance (lowest total mistakes)
- **DeBERTa** was extremely close (difference ~1 sample)
- **DistilBERT** was fastest but had more mistakes

---

## Final Model Choice
✅ **Final selected model: RoBERTa-base**  
Reason: It gave the best test results and is widely accepted in research.

---

## Training Pipeline (A to Z)
This is the full workflow used for training.

### Step 1 — Load dataset
- Load `df_cleaned_verified.csv`
- Keep columns: `text`, `marks`

### Step 2 — Verify labels
- Ensure labels are valid:
  - FAKE → 0
  - REAL → 1
- Remove invalid rows if any

### Step 3 — Clean text
- Remove empty text rows
- Optionally normalize text (lowercasing, removing extra spaces)

### Step 4 — Train/Validation/Test split
Split into:
- Training set
- Validation set
- Test set

A typical split:
- Train: 80%
- Val: 10%
- Test: 10%

### Step 5 — Baseline model
Train a quick baseline such as:
- TF-IDF + Logistic Regression / Linear SVM

This baseline helps prove that the transformer model is truly better.

### Step 6 — Transformer fine-tuning
Fine-tune RoBERTa using Hugging Face `Trainer`:
- Tokenization (max length = 256)
- Dynamic padding
- Train for ~3 epochs
- Evaluate on validation/test

### Step 7 — Save the model
Saving produces:
- `model.safetensors` OR `pytorch_model.bin`
- tokenizer files (`vocab.json`, `merges.txt`, etc.)
- `config.json`

---

## Why the Model is on Hugging Face
GitHub has a file limit:
- **25MB max file size in normal repositories**

Your model files are large:
- `model.safetensors` ≈ 475MB
- `pytorch_model.bin` ≈ 475MB

So the model cannot be pushed to GitHub directly.

✅ Solution: Upload model to **Hugging Face Hub**  
Then Streamlit loads the model from there.

---

## Streamlit App (How it Works)
The Streamlit app is designed like a multi-step system:

### Page 1: Introduction
- Explains the project
- Shows steps
- Button: **Next →** (goes to prediction page)

### Page 2: Predict
- User pastes text input
- Button: **Predict**
- Button: **Result →** (goes to result page)

### Page 3: Result
- Displays:
  - FAKE/REAL decision
  - probabilities for both classes
- Includes Explainable AI charts:
  - token highlights
  - bar chart (top tokens)
  - line chart (attribution shape)

---

## Explainable AI (Integrated Gradients)
The XAI method used is:

✅ **Integrated Gradients (Captum)** on the RoBERTa embedding layer

It provides:
- Token-level importance scores
- Highlights words that influenced the prediction
- Charts to visualize decision patterns

On the Result page, you get:
1. Highlighted tokens (green/red)
2. Bar chart: strongest tokens
3. Line chart: attribution across text index

---

## Run the Streamlit App
[Typical structure:](https://fake-news-detection-7c9ygqy3olqkfqfgdzzuxb.streamlit.app/)

