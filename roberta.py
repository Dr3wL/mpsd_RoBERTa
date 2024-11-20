import os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from tqdm import tqdm
from datetime import datetime

# Paths to directories
malicious_dir = r'C:\Users\Administrator\Desktop\powershell_script_detection\mpsd\malicious_pure'
benign_dir = r'C:\Users\Administrator\Desktop\powershell_script_detection\mpsd\powershell_benign_dataset'
mixed_malicious_dir = r'C:\Users\Administrator\Desktop\powershell_script_detection\mpsd\mixed_malicious'

# File for logging evaluation results
log_file = 'evaluation_results.txt'

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Function to read scripts from a directory and label them
def load_data_from_directories(malicious_dir, benign_dir, mixed_malicious_dir):
    data = []

    # Read benign scripts and label as '0'
    for filename in tqdm(os.listdir(benign_dir), desc="Loading Benign Scripts"):
        file_path = os.path.join(benign_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as script_file:
                script_content = script_file.read()
                data.append({'text': script_content, 'label': 0})
        except Exception as e:
            print(f"Error reading file {filename} in benign directory: {e}")

    # Read malicious scripts and label as '1'
    for filename in tqdm(os.listdir(malicious_dir), desc="Loading Malicious Scripts"):
        file_path = os.path.join(malicious_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as script_file:
                script_content = script_file.read()
                data.append({'text': script_content, 'label': 1})
        except Exception as e:
            print(f"Error reading file {filename} in malicious directory: {e}")

    # Read mixed malicious scripts and label as '1'
    for filename in tqdm(os.listdir(mixed_malicious_dir), desc="Loading Mixed Malicious Scripts"):
        file_path = os.path.join(mixed_malicious_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as script_file:
                script_content = script_file.read()
                # You can change the label to 0 if you believe it's benign
                data.append({'text': script_content, 'label': 1})
        except Exception as e:
            print(f"Error reading file {filename} in mixed malicious directory: {e}")

    return data

# Load dataset
data = load_data_from_directories(malicious_dir, benign_dir, mixed_malicious_dir)

# Convert to Hugging Face's Dataset format
dataset = Dataset.from_list(data)

# Split dataset into training and testing sets (80/20 split)
split_dataset = dataset.train_test_split(test_size=0.2)

# Tokenize dataset
def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)

encoded_dataset = split_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns and set format for PyTorch
encoded_dataset = encoded_dataset.remove_columns(['text'])
encoded_dataset = encoded_dataset.rename_column('label', 'labels')
encoded_dataset.set_format('torch')

# Load pre-trained RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define metrics for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    # Calculate confusion matrix to get FP and TN
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Log the metrics to a file
    with open(log_file, 'a') as log:
        log.write(f"\nEvaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Accuracy: {acc:.4f}\n")
        log.write(f"Precision: {precision:.4f}\n")
        log.write(f"Recall: {recall:.4f}\n")
        log.write(f"F1 Score: {f1:.4f}\n")
        log.write(f"False Positive Rate: {fpr:.4f}\n")
        log.write("-" * 50 + "\n")

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positive_rate': fpr,
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',           # Directory to store model checkpoints
    eval_strategy="epoch",            # Evaluate at the end of each epoch (updated)
    save_strategy="epoch",            # Save the model at the end of each epoch
    learning_rate=2e-5,               # Learning rate
    per_device_train_batch_size=8,    # Training batch size
    per_device_eval_batch_size=16,    # Evaluation batch size
    num_train_epochs=5,               # Number of training epochs
    weight_decay=0.01,                # Strength of weight decay for regularization
    logging_dir='./logs',             # Directory for storing logs
    load_best_model_at_end=True,      # Load the best model at the end of training
    metric_for_best_model='f1',       # Use F1 score to determine the best model
    greater_is_better=True,
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    compute_metrics=compute_metrics,
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-roberta')
tokenizer.save_pretrained('./fine-tuned-roberta')

print(f"Evaluation results have been logged to: {log_file}")
