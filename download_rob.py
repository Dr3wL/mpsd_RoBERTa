from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Download and save the model and tokenizer locally
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Save the model and tokenizer to your local directory
model.save_pretrained('./local_model')
tokenizer.save_pretrained('./local_model')
