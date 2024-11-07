import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer

# Load your fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./fine_tuned_model')
tokenizer = RobertaTokenizer.from_pretrained('./fine_tuned_model')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define a function to summarize code
def summarize_code(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device
    
    # Generate the summary
    summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# Example usage
code_snippet = """
def add(a, b):
    return a + b
"""

print("Code Snippet:")
print(code_snippet)
print("\nGenerated Summary:")
print(summarize_code(code_snippet))
