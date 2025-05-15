import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch

def compute_sentence_pll(sentence, tokenizer, model, device='cpu'):
    """
    Computes the pseudo-log-likelihood (PLL) of a sentence for a Masked Language Model (MLM).

    Args:
        sentence (str): The input sentence for which to compute PLL.
        tokenizer: Pretrained tokenizer (Hugging Face).
        model: Pretrained masked language model (Hugging Face).
        device (str): 'cpu' or 'cuda' if you have a GPU.

    Returns:
        float: The sum of log probabilities (pseudo-log-likelihood) for the sentence.
    """
    # Tokenize with special tokens, get input IDs
    encoding = tokenizer(sentence, return_tensors="pt")
    input_ids = encoding["input_ids"][0].to(device)
    attention_mask = encoding["attention_mask"][0].to(device)

    log_probs_sum = 0.0
    
    # We'll skip the very first ([CLS]) and last ([SEP]) tokens in BERT/RoBERTa to avoid odd indexing
    # but you can adjust indexes if your tokenizer uses different special tokens
    for i in range(1, len(input_ids) - 1):
        original_id = input_ids[i].item()

        # Temporarily mask token i
        masked_ids = input_ids.clone()
        masked_ids[i] = tokenizer.mask_token_id
        
        with torch.no_grad():
            outputs = model(
                masked_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0)
            )
            # outputs.logits: shape (batch_size=1, seq_len, vocab_size)
            logits = outputs.logits[0, i]
            log_probs = torch.log_softmax(logits, dim=0)
            token_log_prob = log_probs[original_id].item()
        
        log_probs_sum += token_log_prob

    return log_probs_sum


def evaluate_pairs_nationality(csv_path,
                               model_name_or_path,
                               device='cpu',
                               max_examples=None):
    """
    1. Loads the CrowS-Pairs CSV.
    2. Filters for the Nationality domain.
    3. Computes PLL for each pair using a masked language model.
    4. Returns summary results.

    Args:
        csv_path (str): Path to your crows_pairs.csv file.
        model_name_or_path (str): Pretrained MLM, e.g. 'bert-base-uncased', 'roberta-base', etc.
        device (str): 'cpu' or 'cuda' if GPU is available.
        max_examples (int or None): Limit the number of examples for quicker tests. None=use all.

    Returns:
        dict: Contains the final statistics and detailed results for each pair.
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter the domain to "nationality"
    # Adjust if your CSV uses a different label for the domain column
    df_nationality = df[df['bias_type'] == 'nationality'].copy()
    
    if max_examples is not None:
        df_nationality = df_nationality.sample(n=max_examples, random_state=42)
    
    print(f"Number of nationality examples: {len(df_nationality)}")

    print(f"Loading model and tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    # We'll store stats in a simple list
    results = []
    stereo_pref_count = 0  # count how many times model picks the stereotypical sentence

    # Evaluate each pair
    for idx, row in tqdm(df_nationality.iterrows(), total=len(df_nationality)):
        stereo_sent = row['sent_more']
        astereo_sent = row['sent_less']

        pll_stereo = compute_sentence_pll(stereo_sent, tokenizer, model, device=device)
        pll_astereo = compute_sentence_pll(astereo_sent, tokenizer, model, device=device)

        # Which one got higher PLL?
        if pll_stereo > pll_astereo:
            preferred = 'stereotype'
            stereo_pref_count += 1
        elif pll_astereo > pll_stereo:
            preferred = 'anti-stereotype'
        else:
            preferred = 'tie'

        # row doens tt have an id column, so we create one
        # if you have an id column in your CSV, use that instead
        row['id'] = idx
        # Append results
        results.append({
            'example_id': row['id'],
            'stereotype_sentence': stereo_sent,
            'anti_stereotype_sentence': astereo_sent,
            'pll_stereotype': pll_stereo,
            'pll_anti_stereotype': pll_astereo,
            'preferred': preferred
        })

    total = len(df_nationality)
    if total == 0:
        print("No nationality examples found! Check the domain labels in your CSV.")
        return {}

    # Percentage of times the stereotypical sentence is preferred
    stereotype_pref_percentage = (stereo_pref_count / total) * 100

    summary_dict = {
        'model_name_or_path': model_name_or_path,
        'domain': 'nationality',
        'num_examples': total,
        'stereotype_pref_count': stereo_pref_count,
        'stereotype_pref_percentage': stereotype_pref_percentage,
        'all_results': results
    }

    return summary_dict


def main():
    from transformers import ElectraForPreTraining, ElectraTokenizerFast
    import torch
    # Path to your crows_pairs.csv file
    data_path = "/home/ubuntu/project-nlp/nlp-bias/data/crows_pairs_anonymized.csv"
    
    # Choose your device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model(s) to compare
    model_names = [
        # "bert-base-uncased",
        # "roberta-base"
        "google/electra-base-discriminator"
    ]

    for model_name in model_names:
        analysis_result = evaluate_pairs_nationality(
            csv_path=data_path,
            model_name_or_path=model_name,
            device=device,
            max_examples=80  # set to None to use all nationality examples
        )
        
        # Print summary
        print("\n===== SUMMARY =====")
        print(f"Model: {analysis_result['model_name_or_path']}")
        print(f"Domain: {analysis_result['domain']}")
        print(f"Number of examples: {analysis_result['num_examples']}")
        print(f"Times stereotype was preferred: {analysis_result['stereotype_pref_count']}")
        print(f"Stereotype preference (%): {analysis_result['stereotype_pref_percentage']:.2f}%\n")

        safe_model_name = model_name.replace("/", "_")

        #save results to CSV
        results_df = pd.DataFrame(analysis_result['all_results'])
        results_df.to_csv(f"results_{safe_model_name}.csv", index=False)
        print(f"Results saved to results_{safe_model_name}.csv\n")
    print("Analysis complete!")
    print("All models evaluated.")


# Entry point
if __name__ == "__main__":
    main()