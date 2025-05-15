# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import RESULTS_DIR
from pathlib import Path

def plot_confusion_matrix(cm, classes, title, filename=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if filename:
        # Create results directory if it doesn't exist
        Path(RESULTS_DIR).mkdir(exist_ok=True)
        plt.savefig(f"{RESULTS_DIR}/{filename}")
    
    plt.show()
    plt.close()

def plot_accuracy_comparison(models, matched_acc, mismatched_acc):
    """Plot accuracy comparison between models"""
    plt.figure(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, matched_acc, width, label='Matched')
    plt.bar(x + width/2, mismatched_acc, width, label='Mismatched')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.legend()
    
    # Add values on bars
    for i, v in enumerate(matched_acc):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
    for i, v in enumerate(mismatched_acc):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    plt.savefig(f"{RESULTS_DIR}/accuracy_comparison.png")
    plt.show()
    plt.close()