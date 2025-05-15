# evaluation.py
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_predictions(predictions, ground_truth):
    """Calculate performance metrics"""
    y_pred = [p['binary_pred'] for p in predictions]
    y_true = ground_truth.tolist()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def display_metrics(metrics, model_name):
    """Print evaluation metrics"""
    print(f"{model_name} Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print()

def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Factual', 'Non-Factual'],
                yticklabels=['Factual', 'Non-Factual'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def display_samples(df, count=5):
    """Display sample predictions"""
    print("\n=== SAMPLE PREDICTIONS ===")
    for idx, row in test_df.sample(5, random_state=42).iterrows():
        premise = row['premise']
        hypothesis = row['hypothesis']
        true_label = row['label']
        
        # Make predictions
        t5_nli, t5_binary = predict_nli_with_t5(premise, hypothesis, *models['t5'])
        gpt2_nli, gpt2_binary = predict_nli_with_gpt2(premise, hypothesis, *models['gpt2'])
        
        # Print detailed example
        print(f"Example {idx}:")
        print(f"Premise: {premise[:150]}..." if len(premise) > 150 else f"Premise: {premise}")
        print(f"Hypothesis: {hypothesis}")
        print(f"True label: {true_label} ({'major_inaccurate' if true_label == 1 else 'accurate'})")
        print(f"T5: {t5_binary} ({t5_nli})")
        print(f"GPT-2: {gpt2_binary} ({gpt2_nli})")
        print("---")

def create_ensemble_prediction(df):
    """Create ensemble prediction using majority voting"""
    df['ensemble_prediction'] = df.apply(
        lambda row: 1 if (row['t5_prediction'] + row['gpt2_prediction'] + row['baseline_prediction'] >= 2) else 0, 
        axis=1
    )
    
    # Print ensemble report
    ensemble_report = classification_report(
        df['binary_label'], 
        df['ensemble_prediction'], 
        target_names=['Factual', 'Non-Factual']
    )
    
    print("\n=== ENSEMBLE MODEL PERFORMANCE ===")
    print(ensemble_report)
    
    # Plot confusion matrix for ensemble
    plot_confusion_matrix(df['binary_label'], df['ensemble_prediction'], 'Ensemble Confusion Matrix')
    
    return df