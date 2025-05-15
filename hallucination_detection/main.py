# main.py - optimized version
import torch
from tqdm import tqdm
import pandas as pd
from time import time

from config import DEVICE, MODEL_CONFIG
from models import predict_nli_with_t5, predict_nli_with_gpt2
from data_utils import load_and_process_data
from evaluation import (
    evaluate_predictions, display_metrics, 
    plot_confusion_matrix, display_samples, create_ensemble_prediction
)

def run_batch_predictions(df, predict_fn, model_name, model, tokenizer, batch_size=8):
    """Run predictions in batches for better efficiency"""
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(df), batch_size), desc=f"{model_name} Predictions"):
        batch = df.iloc[i:i+batch_size]
        batch_results = []
        
        for _, row in batch.iterrows():
            try:
                # Call appropriate prediction function
                if MODEL_CONFIG[model_name.lower()]['simulated'] or model is None or tokenizer is None:
                    # For simulated models, no need to process in parallel
                    prediction = predict_fn(row['premise'], row['hypothesis'], model, tokenizer)
                    batch_results.append(prediction)
                else:
                    # Real models could be optimized further with parallel processing
                    prediction = predict_fn(row['premise'], row['hypothesis'], model, tokenizer)
                    batch_results.append(prediction)
            except Exception as e:
                print(f"Error with {model_name} on row {_}: {e}")
                # Default to contradiction/non-factual for errors
                batch_results.append(("error", 1, str(e)))
        
        # Process batch results
        for idx, (nli_label, binary_pred, extra_info) in zip(batch.index, batch_results):
            result = {
                'idx': idx,
                'nli_label': nli_label,
                'binary_pred': binary_pred
            }
            
            # Add model-specific information
            if model_name.lower() == 't5':
                result['response'] = extra_info
            elif model_name.lower() == 'gpt2':
                result['perplexities'] = extra_info
            else:  # NLI model
                result['probs'] = extra_info
                
            results.append(result)
    
    return results

def main():
    """Main function to run the hallucination detection pipeline"""
    start_time = time()
    print(f"Using device: {DEVICE}")
    
    # Load and process data
    print("Loading and processing data...")
    flat_df = load_and_process_data()
    print(f"Total number of sentences to evaluate: {len(flat_df)}")
    
    # Load models
    print("Loading models...")
    models = load_models()
    
    # Determine batch size based on device and model size
    # Smaller batch size for larger models or CPU
    batch_size = 16 if DEVICE.type == 'cuda' else 4
    if not MODEL_CONFIG['t5']['simulated'] and 'large' in MODEL_CONFIG['t5']['model_name']:
        batch_size = 8  # Smaller batch for large models
    
    print(f"Using batch size: {batch_size}")
    
    # Run predictions with T5
    print("Starting T5 predictions...")
    t5_results = run_batch_predictions(
        flat_df, predict_nli_with_t5, "T5", 
        models['t5'][0], models['t5'][1], batch_size
    )
    
    # Run predictions with GPT-2
    print("Starting GPT-2 predictions...")
    gpt2_results = run_batch_predictions(
        flat_df, predict_nli_with_gpt2, "GPT2", 
        models['gpt2'][0], models['gpt2'][1], batch_size
    )
    
    
    # Evaluate all approaches
    print("Evaluating performance...")
    t5_metrics = evaluate_predictions(t5_results, flat_df['label'])
    gpt2_metrics = evaluate_predictions(gpt2_results, flat_df['label'])
    # baseline_metrics = evaluate_predictions(baseline_results, flat_df['binary_label'])
    
    # Display evaluation results
    print("\n=== EVALUATION RESULTS ===\n")
    model_name = "Flan-T5-small" if "flan-t5-small" in MODEL_CONFIG['t5']['model_name'] else "T5"
    display_metrics(t5_metrics, f"{model_name} NLI Model")
    display_metrics(gpt2_metrics, "GPT-2 NLI Model")
    # display_metrics(baseline_metrics, "Baseline Pre-trained NLI Model")
    
    # Add predictions to dataframe for analysis
    flat_df['t5_prediction'] = [r['binary_pred'] for r in t5_results]
    flat_df['t5_nli_label'] = [r['nli_label'] for r in t5_results]
    
    flat_df['gpt2_prediction'] = [r['binary_pred'] for r in gpt2_results]
    flat_df['gpt2_nli_label'] = [r['nli_label'] for r in gpt2_results]
    
    # flat_df['baseline_prediction'] = [r['binary_pred'] for r in baseline_results]
    # flat_df['baseline_nli_label'] = [r['nli_label'] for r in baseline_results]
    
    # Plot confusion matrices
    plot_confusion_matrix(flat_df['binary_label'], flat_df['t5_prediction'], f'{model_name} Confusion Matrix')
    plot_confusion_matrix(flat_df['binary_label'], flat_df['gpt2_prediction'], 'GPT-2 Confusion Matrix')
    # plot_confusion_matrix(flat_df['binary_label'], flat_df['baseline_prediction'], 'Baseline Confusion Matrix')
    
    # Create and evaluate ensemble model
    flat_df = create_ensemble_prediction(flat_df)
    
    # Display sample predictions
    display_samples(flat_df)
    
    # Save results to CSV
    result_file = 'hallucination_detection_results.csv'
    flat_df.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")
    
    # Calculate and display total runtime
    runtime = time() - start_time
    print(f"\nTotal runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")

if __name__ == "__main__":
    main()