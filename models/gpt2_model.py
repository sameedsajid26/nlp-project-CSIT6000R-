# models/gpt2_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base_model import NLIModel
from config import DEVICE, MAX_NEW_TOKENS, TEMPERATURE

class GPT2Model(NLIModel):
    """GPT-2 model for NLI"""
    
    def __init__(self, model_name="gpt2"):
        super().__init__(model_name)
        self.verbalizer = {
            "entailment": ["entailment", "entail", "true", "correct", "yes"],
            "contradiction": ["contradiction", "contradict", "false", "incorrect", "no"],
            "neutral": ["neutral", "maybe", "unclear", "neither", "possible"]
        }
    
    def load(self):
        """Load the model and tokenizer"""
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(DEVICE)
        return self
    
    def predict(self, premise, hypothesis, prompt_fn):
        """Make a prediction for a single example"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        prompt = prompt_fn(premise, hypothesis)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Get the generated text
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        prediction_text = generated_text[len(prompt):].strip().lower()
        
        # Determine the label based on the verbalizer
        for label, tokens in self.verbalizer.items():
            if any(token in prediction_text for token in tokens):
                return label
        
        # Default to most frequent class if no match
        return "neutral"