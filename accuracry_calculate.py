import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Output file with predictions
output_file = 'pretrained_model_results.txt'

# Initialize lists to track true and predicted labels globally and per directory
true_labels = []
predicted_labels = []

# Per-directory metrics
metrics_per_directory = {
    'malicious_pure': {'true': [], 'predicted': []},
    'powershell_benign_dataset': {'true': [], 'predicted': []},
    'mixed_malicious': {'true': [], 'predicted': []}
}

# Track current directory context
current_directory = None

# Read the output file and parse predictions
with open(output_file, 'r') as f:
    for line in f:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        

        # Detect directory changes in the output file
        if "Directory:" in line:
            directory_path = line.split("Directory: ")[1].strip()
            current_directory = os.path.basename(directory_path)
            print(f"DEBUG: Changing directory context to: {current_directory}")  # Debugging: Directory detected

        # Check if the line contains a script label
        elif "Script:" in line and "Label:" in line and current_directory:

            # Extract the predicted label from the line
            parts = line.split(", ")
            filename = parts[0].split(": ")[1]
            pred_label_str = parts[1].split(": ")[1]
            pred_label = 1 if pred_label_str == 'Malicious' else 0


            # Assign true label based on the current directory context
            if current_directory == 'malicious_pure':
                true_label = 1  # Malicious
                metrics_per_directory['malicious_pure']['true'].append(1)
                metrics_per_directory['malicious_pure']['predicted'].append(pred_label)
            elif current_directory == 'powershell_benign_dataset':
                true_label = 0  # Benign
                metrics_per_directory['powershell_benign_dataset']['true'].append(0)
                metrics_per_directory['powershell_benign_dataset']['predicted'].append(pred_label)
            elif current_directory == 'mixed_malicious':
                true_label = 1  # Malicious
                metrics_per_directory['mixed_malicious']['true'].append(1)
                metrics_per_directory['mixed_malicious']['predicted'].append(pred_label)
            else:
                print(f"DEBUG: Unrecognized directory: {current_directory}")
                continue

            # Append to global lists
            true_labels.append(true_label)
            predicted_labels.append(pred_label)

# Check if we have valid data to compute metrics
if true_labels and predicted_labels:
    # Calculate global accuracy and other metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')

    # Print global accuracy and metrics
    print("=== Global Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
else:
    print("No valid data to compute global metrics.")

# Function to print per-directory statistics
def print_metrics_per_directory(directory_name, true, predicted):
    if true and predicted:  # Avoid empty data lists
        accuracy = accuracy_score(true, predicted)
        precision, recall, f1, _ = precision_recall_fscore_support(true, predicted, average='binary')
        print(f"\n=== Metrics for {directory_name} ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    else:
        print(f"\n=== No data for {directory_name} ===")

# Print metrics for each directory
print_metrics_per_directory('Malicious Pure', metrics_per_directory['malicious_pure']['true'], metrics_per_directory['malicious_pure']['predicted'])
print_metrics_per_directory('Benign Dataset', metrics_per_directory['powershell_benign_dataset']['true'], metrics_per_directory['powershell_benign_dataset']['predicted'])
print_metrics_per_directory('Mixed Malicious', metrics_per_directory['mixed_malicious']['true'], metrics_per_directory['mixed_malicious']['predicted'])
