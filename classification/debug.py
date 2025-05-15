# debug_dataset.py
import json
import pandas as pd
from datasets import Dataset
from config import MATCHED_DATA_PATH, MISMATCHED_DATA_PATH

def examine_jsonl(file_path, num_examples=3):
    """Directly examine JSONL file contents"""
    print(f"\nExamining {file_path}:")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_examples:
                    break
                try:
                    data = json.loads(line)
                    print(f"Example {i+1}:")
                    print(f"  Type: {type(data)}")
                    print(f"  Keys: {list(data.keys())}")
                    if 'sentence1' in data:
                        print(f"  Premise: {data['sentence1'][:50]}...")
                    if 'sentence2' in data:
                        print(f"  Hypothesis: {data['sentence2'][:50]}...")
                    if 'gold_label' in data:
                        print(f"  Label: {data['gold_label']}")
                    print()
                except Exception as e:
                    print(f"  Error parsing line {i+1}: {e}")
    except Exception as e:
        print(f"  Error opening file: {e}")

def test_dataset_access(file_path):
    """Test different ways to access the dataset"""
    print(f"\nTesting dataset access for {file_path}:")
    
    # Load as DataFrame first
    df = pd.read_json(file_path, lines=True)
    print(f"DataFrame loaded, shape: {df.shape}")
    
    # Convert to Dataset
    dataset = Dataset.from_pandas(df)
    print(f"Dataset created, length: {len(dataset)}")
    
    # Test accessing first item
    if len(dataset) > 0:
        first_item = dataset[0]
        print(f"First item type: {type(first_item)}")
        
        # Try different access methods
        try:
            print("Dictionary-style access:")
            print(f"  sentence1: {first_item['sentence1'][:50]}...")
            print(f"  sentence2: {first_item['sentence2'][:50]}...")
            print(f"  gold_label: {first_item['gold_label']}")
        except Exception as e:
            print(f"  Error with dictionary access: {e}")
        
        # Try attribute access
        try:
            print("\nAttribute-style access:")
            print(f"  sentence1: {first_item.sentence1[:50]}...")
            print(f"  sentence2: {first_item.sentence2[:50]}...")
            print(f"  gold_label: {first_item.gold_label}")
        except Exception as e:
            print(f"  Error with attribute access: {e}")
        
        # Try to get all attributes
        try:
            print("\nAll available attributes:")
            print(f"  dir(first_item): {dir(first_item)}")
        except Exception as e:
            print(f"  Error getting attributes: {e}")
    
    # Test batch access
    print("\nTesting batch access:")
    try:
        batch = dataset[:2]
        print(f"Batch type: {type(batch)}")
        print(f"Batch fields: {list(batch.keys()) if hasattr(batch, 'keys') else 'No keys method'}")
        
        # Try to access sentences in batch
        if hasattr(batch, 'keys') and 'sentence1' in batch:
            print(f"First premise in batch: {batch['sentence1'][0][:50]}...")
        else:
            print("Cannot access 'sentence1' directly from batch")
    except Exception as e:
        print(f"Error with batch access: {e}")

def main():
    print("=== Dataset Debug Tool ===")
    
    # Examine raw JSONL files
    examine_jsonl(MATCHED_DATA_PATH)
    examine_jsonl(MISMATCHED_DATA_PATH)
    
    # Test dataset access methods
    test_dataset_access(MATCHED_DATA_PATH)
    test_dataset_access(MISMATCHED_DATA_PATH)

if __name__ == "__main__":
    main()