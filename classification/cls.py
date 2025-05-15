import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import torch.nn.functional as F

# Load the dataset
dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation")

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame({
    'wiki_bio_text': dataset['wiki_bio_text'],
    'gpt3_sentences': dataset['gpt3_sentences'],
    'annotations': dataset['annotation']
})

# print(df.columns)

# Flatten the data so each row is a single sentence with its annotation
flattened_data = []
for idx, row in df.iterrows():
    wiki_text = row['wiki_bio_text']
    for sent, annot in zip(row['gpt3_sentences'], row['annotations']):
        # Convert to binary classification: 0 for Factual, 1 for Non-Factual (both minor and major)
        binary_label = 0 if annot == "accurate" else 1
        flattened_data.append({
            'premise': wiki_text,
            'hypothesis': sent,
            'original_label': annot,
            'binary_label': binary_label
        })

flat_df = pd.DataFrame(flattened_data)
print(f"Total number of sentences to evaluate: {len(flat_df)}")
print(flat_df.head())