import os
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from datetime import datetime
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from collections import Counter


# Paths to directories
malicious_dir = r'E:/Capstone/mpsd_RoBERTa/malicious_pure'
benign_dir = r'E:/Capstone/mpsd_RoBERTa/powershell_benign_dataset'
mixed_malicious_dir = r'E:/Capstone/mpsd_RoBERTa/mixed_malicious'

# File for logging evaluation results
log_file = 'evaluation_results_NO_Mixed.txt'
keyword_log_file = 'keyword_frequencies_NO_mixed.txt'

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Advanced feature extraction functions
def count_keywords(script_content):
    keywords = [
        "Invoke-WebRequest", "Invoke-Expression", "-EncodedCommand", "-NoProfile",
        "DownloadString", "Add-Member", "New-Object", "Set-ExecutionPolicy"
    ]
    return {kw: script_content.count(kw) for kw in keywords}

def count_parameters(script_content):
    parameters = ["-NoProfile", "-WindowStyle", "-EncodedCommand", "-Command"]
    return {param: script_content.count(param) for param in parameters}

def calculate_function_call_depth(script_content):
    return script_content.count("{") - script_content.count("}")

def count_obfuscation_patterns(script_content):
    patterns = ["$$", "--", "`", "^", "|"]
    return {pattern: script_content.count(pattern) for pattern in patterns}

def extract_weighted_cooccurrences(script_content, cooccurrence_weights):
    cooccurrence_features = {}
    for (kw1, kw2), weight in cooccurrence_weights.items():
        feature_name = f"{kw1}_and_{kw2}"
        cooccurrence_features[feature_name] = int(kw1 in script_content and kw2 in script_content) * weight
    return cooccurrence_features

def calculate_keyword_proximity(script_content, keyword_pairs):
    proximity_features = {}
    words = script_content.split()
    for kw1, kw2 in keyword_pairs:
        indices_kw1 = [i for i, word in enumerate(words) if word == kw1]
        indices_kw2 = [i for i, word in enumerate(words) if word == kw2]
        if indices_kw1 and indices_kw2:
            proximity = min([abs(i - j) for i in indices_kw1 for j in indices_kw2])
        else:
            proximity = len(words)  # Assign max distance if one keyword is missing
        proximity_features[f"{kw1}_to_{kw2}_proximity"] = proximity
    return proximity_features

# Co-occurrence weights and proximity pairs
cooccurrence_weights = {
    ("Invoke-Expression", "New-Object"): 3,
    ("DownloadString", "New-Object"): 2.5,
    ("Invoke-Expression", "-NoProfile"): 2,
    ("-NoProfile", "New-Object"): 2,
}
keyword_pairs = [("Invoke-WebRequest", "DownloadString"), ("Invoke-Expression", "Set-ExecutionPolicy")]

# Preprocess script and extract features
def preprocess_powershell_script(script_content, label):
    keyword_counts = count_keywords(script_content)
    parameter_counts = count_parameters(script_content)
    function_depth = calculate_function_call_depth(script_content)
    obfuscation_counts = count_obfuscation_patterns(script_content)
    cooccurrence_features = extract_weighted_cooccurrences(script_content, cooccurrence_weights)
    proximity_features = calculate_keyword_proximity(script_content, keyword_pairs)

    features = {
        **keyword_counts,
        **parameter_counts,
        'function_call_depth': function_depth,
        **obfuscation_counts,
        **cooccurrence_features,
        **proximity_features
    }
    return {'text': script_content, 'features': list(features.values()), 'label': label}

# Log keyword frequencies
def log_keyword_frequencies(data):
    keyword_counter = Counter()
    for example in data:
        script_content = example['text']
        keyword_counter.update(count_keywords(script_content))
    
    with open(keyword_log_file, 'w', encoding='utf-8') as log:
        log.write("Keyword Frequencies:\n")
        for keyword, freq in keyword_counter.most_common():
            log.write(f"{keyword}: {freq}\n")

# Load data from directories
def load_data_from_directories(malicious_dir, benign_dir):
    data = []
    for filename in tqdm(os.listdir(benign_dir), desc="Loading Benign Scripts"):
        file_path = os.path.join(benign_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as script_file:
            script_content = script_file.read()
            data.append(preprocess_powershell_script(script_content, 0))
    for filename in tqdm(os.listdir(malicious_dir), desc="Loading Malicious Scripts"):
        file_path = os.path.join(malicious_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as script_file:
            script_content = script_file.read()
            data.append(preprocess_powershell_script(script_content, 1))
    return data


# Tokenize and encode features
def tokenize_and_encode_features(example):
    tokens = tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)
    tokens['features'] = torch.tensor(example['features'], dtype=torch.float32)  # Convert to float32
    tokens['labels'] = example['label']
    return tokens

# Define the hybrid model
class HybridRobertaModel(nn.Module):
    def __init__(self, feature_dim, num_labels=2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.feature_dense = nn.Linear(feature_dim, self.roberta.config.hidden_size)
        self.classifier = nn.Linear(self.roberta.config.hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, features, labels=None):
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = roberta_outputs.last_hidden_state[:, 0, :]
        feature_embeddings = self.feature_dense(features.float())
        combined_embeddings = torch.cat((text_embeddings, feature_embeddings), dim=1)
        logits = self.classifier(combined_embeddings)

        loss = None
        if labels is not None:
            class_weights = torch.tensor([1.0, 1.5], dtype=torch.float).to(input_ids.device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

# Load and preprocess dataset
prepared_dataset_path = './prepared_dataset'
if not os.path.exists(prepared_dataset_path):
    data = load_data_from_directories(malicious_dir, benign_dir)
    feature_dim = len(data[0]['features'])  # Compute feature_dim here
    dataset = Dataset.from_list(data)
    split_dataset = dataset.train_test_split(test_size=0.2)
    encoded_dataset = split_dataset.map(tokenize_and_encode_features, batched=True)
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'features', 'labels'])
    encoded_dataset.save_to_disk(prepared_dataset_path)
else:
    from datasets import load_from_disk
    encoded_dataset = load_from_disk(prepared_dataset_path)
    # Calculate feature_dim from one example in the loaded dataset
    feature_dim = len(encoded_dataset['train'][0]['features'])  # Adjust if necessary

# Initialize hybrid model
model = HybridRobertaModel(feature_dim=feature_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Save predictions for further analysis
def log_predictions(predictions, labels):
    preds = predictions.argmax(-1)
    with open("predictions_and_labels.txt", "w", encoding="utf-8") as log:
        log.write("Predicted,True\n")
        for pred, label in zip(preds, labels):
            log.write(f"{pred},{label}\n")

# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Confusion Matrix Components
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # Metrics Calculation
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    # False Positive Rate and True Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Avoid division by zero
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall is equivalent to TPR

    # Log results to evaluation_results.txt
    with open(log_file, 'a') as log:
        log.write(f"\nEvaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall/TPR: {recall:.4f}, F1: {f1:.4f}, FPR: {fpr:.4f}\n")
        log.write("-" * 50 + "\n")

    # Log predictions for further analysis
    log_predictions(predictions=pred.predictions, labels=labels)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positive_rate': fpr,
        'true_positive_rate': tpr
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()
