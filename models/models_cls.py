import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.base_model import NLIModel
from config import DEVICE, MAX_NEW_TOKENS, TEMPERATURE
from transformers import T5ForConditionalGeneration, AutoTokenizer
from config import DEVICE, MAX_NEW_TOKENS



# Function for T5 prompting-based NLI
def predict_nli_with_t5_prompting(premise, hypothesis, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Format prompt for T5
    prompt = f"Given the following information: '{premise}' Does this statement follow: '{hypothesis}'? Answer with 'entailment', 'neutral', or 'contradiction'."
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=50,
            num_beams=4,
            early_stopping=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    
    # Extract NLI label from response
    if "entailment" in response:
        nli_label = "entailment"
        hallucination_score = 0  # Factual
    elif "contradiction" in response:
        nli_label = "contradiction"
        hallucination_score = 1  # Non-Factual
    else:  # neutral or unclear
        nli_label = "neutral"
        hallucination_score = 0.5  # Uncertain, could go either way
    
    # For binary classification, we'll set a threshold on the hallucination score
    binary_pred = 1 if hallucination_score > 0.25 else 0
    
    return nli_label, binary_pred, response


# Function for GPT-2 prompting-based NLI
def predict_nli_with_gpt2_prompting(premise, hypothesis, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
    # For GPT-2, we'll construct prompts for each label and compare likelihood
    prompts = {
        "entailment": f"Given that {premise} Therefore, {hypothesis}",
        "neutral": f"Given that {premise} It's unclear whether {hypothesis}",
        "contradiction": f"Given that {premise} However, {hypothesis} is false."
    }
    
    # Calculate perplexity for each prompt
    perplexities = {}
    for label, prompt in prompts.items():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(inputs.input_ids, labels=inputs.input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        perplexities[label] = perplexity
    
    # Find the label with the lowest perplexity
    nli_label = min(perplexities, key=perplexities.get)
    
    # Map to hallucination detection
    if nli_label == "entailment":
        hallucination_score = 0  # Factual
    elif nli_label == "contradiction":
        hallucination_score = 1  # Non-Factual
    else:  # neutral
        # For neutral, check if it's closer to contradiction or entailment
        if perplexities["contradiction"] < perplexities["entailment"]:
            hallucination_score = 0.75  # Leaning towards non-factual
        else:
            hallucination_score = 0.25  # Leaning towards factual
    
    binary_pred = 1 if hallucination_score > 0.5 else 0
    
    return nli_label, binary_pred, perplexities

    # Function for pre-trained NLI model prediction
def predict_with_pretrained_nli(premise, hypothesis, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    
    # Map to labels (typically: 0=entailment, 1=neutral, 2=contradiction for RoBERTa-MNLI)
    labels = ["entailment", "neutral", "contradiction"]
    nli_label = labels[pred_idx]
    
    # Map to binary for hallucination
    if nli_label == "entailment":
        hallucination_score = 0  # Factual
    elif nli_label == "contradiction":
        hallucination_score = 1  # Non-Factual
    else:  # neutral
        # For neutral, use probability distribution to make a decision
        entail_prob = probs[0].item()
        contra_prob = probs[2].item()
        
        if contra_prob > entail_prob:
            hallucination_score = 0.75  # Leaning towards non-factual
        else:
            hallucination_score = 0.25  # Leaning towards factual
    
    binary_pred = 1 if hallucination_score > 0.5 else 0
    
    return nli_label, binary_pred, probs.tolist()