# config.py
import torch

# Model configuration
MODEL_CONFIG = {
    't5': {
        'model_name': 'google/flan-t5-small',  # Use a smaller model for simulation
        'simulated': False,  # Set to False to use actual T5 model
    },
    'gpt2': {
        'model_name': 'gpt2',
        'simulated': False,  # Set to False to use actual GPT-2 model
    },
    'nli': {
        'model_name': 'roberta-large-mnli',
        'simulated': False,  # We'll use the actual NLI model
    }
}

# Data configuration
DATA_CONFIG = {
    'dataset_name': 'potsawee/wiki_bio_gpt3_hallucination',
    'split': 'evaluation',
    'sample_size': 50,  # Set to None to use full dataset
    'random_state': 42,
}

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')