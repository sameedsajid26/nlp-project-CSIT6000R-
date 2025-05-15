# config.py
import torch

# Paths
MATCHED_DATA_PATH = "Dataset/dev_matched_sampled-1.jsonl"
MISMATCHED_DATA_PATH = "Dataset/dev_mismatched_sampled-1.jsonl"
RESULTS_DIR = "results"

# Model configs
GPT2_MODEL_NAME = "gpt2"  # Use "gpt2-medium" for better results
T5_MODEL_NAME = "google/flan-t5-small"  # Use "google/flan-t5-base" for better results

# Runtime configs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 8
NUM_TEST_SAMPLES = 10  # Number of samples to test with

# Prompt type
PROMPT_TYPE = "few_shot"  # Options: "zero_shot", "few_shot", "cot"

# Generation parameters
MAX_NEW_TOKENS = 20
TEMPERATURE = 0.7