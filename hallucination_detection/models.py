# models.py
import random
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_CONFIG, DEVICE

def predict_nli_with_t5(premise, hypothesis, model=None, tokenizer=None):
    """Predict NLI label using T5 model"""
    if model is None or tokenizer is None:
        # Simulated version using word overlap
        return simulate_nli_prediction(premise, hypothesis)
    
    # Format the input as an NLI task
    prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? Answer with 'entailment', 'neutral', or 'contradiction'."
    
    # Print prompt for debugging in a few cases
    if random.random() < 0.05:  # Print 5% of prompts for debugging
        print(f"\n=== T5 PROMPT ===\n{prompt}\n================")
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    
    # Generate the answer
    outputs = model.generate(
        inputs.input_ids, 
        max_length=20, 
        num_beams=4, 
        early_stopping=True
    )
    
    # Decode the prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    
    # Print response for debugging in a few cases
    if random.random() < 0.05:
        print(f"T5 Response: '{prediction}'")
    
    # Map the prediction to NLI label
    if "entailment" in prediction or "yes" in prediction:
        nli_label = "entailment"
    elif "contradiction" in prediction or "no" in prediction:
        nli_label = "contradiction"
    else:  # neutral or unclear
        nli_label = "neutral"
    
    # Map NLI label to binary factuality prediction
    if nli_label == "entailment":
        binary_pred = 0  # Factual
    elif nli_label == "contradiction":
        binary_pred = 1  # Non-Factual (hallucination)
    else:  # neutral
        # For neutral cases, lean toward factual (this is a design choice)
        binary_pred = 0
    
    return nli_label, binary_pred

def predict_nli_with_gpt2(premise, hypothesis, model=None, tokenizer=None):
    """Predict NLI label using GPT-2 perplexity"""
    if model is None or tokenizer is None:
        # Simulated version using word overlap
        return simulate_nli_prediction(premise, hypothesis)
    
    # Format prompts for different NLI relationships
    prompts = {
        "entailment": f"{premise} Therefore, {hypothesis}",
        "neutral": f"{premise} It's possible that {hypothesis}",
        "contradiction": f"{premise} However, {hypothesis} is false."
    }
    
    # Calculate perplexity for each prompt
    perplexities = {}
    
    # Debug counter to print some examples
    debug_this = random.random() < 0.05  # Debug 5% of examples
    
    if debug_this:
        print("\n=== GPT-2 DEBUG ===")
        print(f"Premise: {premise[:100]}...")
        print(f"Hypothesis: {hypothesis}")
    
    for label, prompt in prompts.items():
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Handle long inputs
        if inputs.input_ids.shape[1] > 1024:
            if debug_this:
                print(f"Truncating input from {inputs.input_ids.shape[1]} tokens")
            inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
        
        # Calculate loss (log likelihood)
        with torch.no_grad():
            outputs = model(inputs.input_ids, labels=inputs.input_ids)
            neg_log_likelihood = outputs.loss.item()
            
            # Perplexity = exp(average negative log likelihood)
            perplexity = torch.exp(torch.tensor(neg_log_likelihood)).item()
            
            # Store perplexity
            perplexities[label] = perplexity
            
            if debug_this:
                print(f"{label} perplexity: {perplexity}")
    
    # Select label with lowest perplexity
    nli_label = min(perplexities, key=perplexities.get)
    
    # Map NLI label to binary prediction
    if nli_label == "entailment":
        binary_pred = 0  # Factual
    elif nli_label == "contradiction":
        binary_pred = 1  # Non-Factual (hallucination)
    else:  # neutral
        # For neutral, compare entailment vs contradiction perplexities
        ent_perp = perplexities["entailment"]
        con_perp = perplexities["contradiction"]
        binary_pred = 0 if ent_perp <= con_perp else 1
    
    if debug_this:
        print(f"Selected label: {nli_label}")
        print(f"Binary prediction: {binary_pred}")
        print("===================")
    
    return nli_label, binary_pred

# Add simulate_nli_prediction as a fallback
def simulate_nli_prediction(premise, hypothesis):
    """Simulate NLI prediction based on word overlap when models aren't available"""
    # Simple word overlap-based heuristic
    premise_words = set(premise.lower().split())
    hypothesis_words = set(hypothesis.lower().split())
    
    # Calculate word overlap ratio
    overlap = len(premise_words.intersection(hypothesis_words)) / len(hypothesis_words) if hypothesis_words else 0
    
    # Determine NLI label based on overlap ratio
    if overlap > 0.7:
        nli_label = "entailment"
        binary_pred = 0  # Factual
    elif overlap < 0.3:
        nli_label = "contradiction"
        binary_pred = 1  # Non-Factual
    else:
        nli_label = "neutral"
        binary_pred = 0  # Default to factual for neutral
    
    return nli_label, binary_pred