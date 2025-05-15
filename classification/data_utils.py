# data_utils.py
import json
import pandas as pd
import numpy as np
from datasets import Dataset
import random
from config import SEED, NUM_TEST_SAMPLES

random.seed(SEED)

def load_jsonl_safe(file_path):
    """Load JSONL file with error handling"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {e}")
                    continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None
    return pd.DataFrame(data)

def load_dataset(file_path, sample=False):
    """Load dataset and optionally return a small sample"""
    df = load_jsonl_safe(file_path)
    if df is None:
        raise ValueError(f"Failed to load dataset from {file_path}")
    
    # Handle incorrect label format if needed
    if '-' in df['gold_label'].values:
        print("Warning: Found '-' in gold_label. Replacing with 'neutral'")
        df['gold_label'] = df['gold_label'].replace('-', 'neutral')
    
    # Sample the data if requested
    if sample:
        if len(df) > NUM_TEST_SAMPLES:
            df = df.sample(NUM_TEST_SAMPLES, random_state=SEED)
    
    # Convert to dataset
    dataset = Dataset.from_pandas(df)
    
    return dataset

def get_label_distribution(dataset):
    """Get the distribution of labels in the dataset"""
    labels = []
    for i in range(len(dataset)):
        try:
            labels.append(dataset[i]["gold_label"])
        except:
            pass
    unique_labels, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique_labels, counts))

def print_dataset_info(name, dataset):
    """Print information about the dataset"""
    print(f"\n{name} Dataset:")
    print(f"  Number of samples: {len(dataset)}")
    
    # Print label distribution
    label_dist = get_label_distribution(dataset)
    print(f"  Label distribution:")
    for label, count in label_dist.items():
        print(f"    {label}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Print a sample example
    if len(dataset) > 0:
        print("\n  Sample example:")
        example = dataset[0]
        print(f"    Premise: {example['sentence1']}")
        print(f"    Hypothesis: {example['sentence2']}")
        print(f"    Gold label: {example['gold_label']}")