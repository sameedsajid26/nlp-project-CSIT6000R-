# models/t5_model.py
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from models.base_model import NLIModel
from config import DEVICE, MAX_NEW_TOKENS

class T5Model(NLIModel):
    """T5 model for NLI"""
    
    def __init__(self, model_name="google/flan-t5-small"):
        super().__init__(model_name)
        self.verbalizer = {
            "entailment": ["entailment", "entail", "yes", "true"],
            "contradiction": ["contradiction", "contradict", "no", "false"],
            "neutral": ["neutral", "neither", "maybe", "unclear", "unknown"]
        }
    
    def load(self):
        """Load the model and tokenizer"""
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(DEVICE)
        return self
    
    def predict(self, premise, hypothesis, prompt_fn):
        """Make a prediction for a single example"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        try:
            # Create the prompt
            prompt = prompt_fn(premise, hypothesis)
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=MAX_NEW_TOKENS,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode the generated text
            prediction_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            
            # Map to label using verbalizer
            for label, tokens in self.verbalizer.items():
                if any(token in prediction_text for token in tokens):
                    return label
        
        except Exception as e:
            print(f"Error during T5 prediction: {e}")
        
        # Default to neutral if no match or error
        return "neutral"