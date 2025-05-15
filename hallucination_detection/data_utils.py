# data_utils.py (revised)
import pandas as pd
from datasets import load_dataset
from config import DATA_CONFIG

def load_and_process_data():
    """Load and process the dataset"""
    # Print raw sample to understand the dataset structure
    dataset = load_dataset(DATA_CONFIG['dataset_name'], split=DATA_CONFIG['split'])
    
    # Print the first example to understand structure
    print("=== DATASET STRUCTURE ===")
    print(dataset[0])
    print("=========================")
    
    if DATA_CONFIG['sample_size']:
        dataset = dataset.shuffle(seed=DATA_CONFIG['random_state']).select(range(DATA_CONFIG['sample_size']))
    
    # Create a flat dataframe with all premise-hypothesis pairs
    flat_data = []
    for item in dataset:
        # Based on the dataset structure, 'wiki_bio_text' is the premise and 'gpt3_sentences' contains hypotheses
        premise = item['wiki_bio_text']
        
        # For each sentence in 'gpt3_sentences', create a separate data point
        for i, hypothesis in enumerate(item['gpt3_sentences']):
            # Get corresponding label (if available)
            if 'annotation' in item and i < len(item['annotation']):
                label_str = item['annotation'][i]
                label = 1 if label_str == 'major_inaccurate' else 0
            else:
                # Default to 0 if label is not available
                label = 0
            
            flat_data.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'label': label,
                'original_label': label_str if 'annotation' in item and i < len(item['annotation']) else None
            })
    
    return pd.DataFrame(flat_data)