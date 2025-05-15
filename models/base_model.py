# models/base_model.py
from abc import ABC, abstractmethod
from tqdm import tqdm

class NLIModel(ABC):
    """Abstract base class for NLI models"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    def load(self):
        """Load the model and tokenizer"""
        pass
    
    @abstractmethod
    def predict(self, premise, hypothesis, prompt_fn):
        """Make a prediction for a single example"""
        pass
    
    def predict_batch(self, dataset, prompt_fn, batch_size=8, max_samples=None):
        """Make predictions for a batch of examples"""
        # Limit the number of samples if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
        
        predictions = []
        # Process the dataset one item at a time
        for i in tqdm(range(0, len(dataset)), desc=f"Predicting with {self.model_name}"):
            try:
                # Access each item individually
                example = dataset[i]
                
                # Extract premise and hypothesis
                premise = example["sentence1"]
                hypothesis = example["sentence2"]
                
                # Make prediction
                pred = self.predict(premise, hypothesis, prompt_fn)
                predictions.append(pred)
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                # Default prediction
                predictions.append("neutral")
                
        return predictions
    
    def __str__(self):
        return self.model_name