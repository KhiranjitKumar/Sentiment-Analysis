# Sentiment Classification using BERT-BiLSTM

## Project Overview

This project is a binary sentiment classification task that predicts the polarity (positive or negative) of movie reviews. We use a **BERT** (Bidirectional Encoder Representations from Transformers) model combined with a **BiLSTM** (Bidirectional Long Short-Term Memory) network for this task.

## Project Structure

The project follows this directory structure:

```
sentiment-classification-NLP/
│
├── data/               # Directory for raw dataset (not uploaded)
├── logs/               # Directory for model logs and saved checkpoints
├── scripts/            # Directory for Python scripts
│   ├── preprocess.py   # Preprocessing and data tokenization
│   ├── model.py        # BERT-BiLSTM model definition
│   ├── train.py        # Model training script
│   ├── test.py         # Model testing and evaluation script
├── README.md           # This file
├── .gitignore          # File specifying files/folders to ignore

```

The fle create_project.py creates the folder structure and neccessary python fles.

## Requirements

- python
- PyTorch
- Transformers (Hugging Face)
- (Hugging Face)
- Scikit-learn

## Dataset

The dataset is from (https://www.cs.cornell.edu/people/pabo/movie-review-data/). Place the files (`rt-polarity.pos` and `rt-polarity.neg`) in the `data/` directory.

## How to Run the Project

### 1. Data Preprocessing

To preprocess and tokenize the dataset:

```bash
python scripts/preprocess.py
```

This script will load, split, and tokenize the dataset using a pre-trained BERT tokenizer.

### 2. Training the Model

Build the nodel using

```bash
python scripts/model.py
```

To train the BERT-BiLSTM model:

```bash
python scripts/train.py
```

The best model based on validation loss will be saved in the `logs/` directory.

### 3. Testing the Model

Once the model is trained, evaluate it on the train dataset:

```bash
python scripts/evaluate.py
```

Test the model on unseen test data by

```bash
python scripts/test.py
```

The scripts will output the confusion matrix, precision, recall, and F1 score.

## Model Architecture

The model combines BERT with a BiLSTM layer:

1. **BERT**: Extracts contextual embeddings from sentences.
2. **BiLSTM**: Captures long-term dependencies from the BERT embeddings.
3. **Fully Connected Layer**: Produces binary classification output (positive/negative).

## Results

After training, the model achieves the following metrics on the train set:

- **Precision**: 0.8906
- **Recall**: 0.9380
- **F1 Score**:0.9137
- **confusion Matrix**:

```bash
True Positives (TP): 3752
True Negatives (TN): 3539
False Positives (FP): 461
False Negatives (FN): 248
```

The model achieves the following metrics on the test set:

- **Precision**: 0.8885
- **Recall**: 0.9398
- **F1 Score**:00.9135
- **confusion Matrix**:

```bash
True Positives (TP): 781
True Negatives (TN): 733
False Positives (FP): 98
False Negatives (FN): 50
```
