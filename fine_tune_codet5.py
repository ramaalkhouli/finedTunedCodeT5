import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load dataset
dataset = load_dataset('code_x_glue_ct_code_to_text', 'python')


# Preprocessing function
def preprocess_function(examples):
    inputs = examples['code']
    targets = examples['docstring']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set up training arguments for Colab (uses GPU if available)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    report_to="none"  # Disabling reporting to prevent warnings
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# Train and evaluate the model
trainer.train()
trainer.evaluate(eval_dataset=tokenized_datasets['test'])

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
