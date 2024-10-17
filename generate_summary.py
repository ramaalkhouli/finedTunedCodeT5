from transformers import T5ForConditionalGeneration, RobertaTokenizer

# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./fine_tuned_model')
tokenizer = RobertaTokenizer.from_pretrained('./fine_tuned_model')

# Read the Python file (test.py)
with open('test.py', 'r') as file:
    code = file.read()

# Tokenize the input code
inputs = tokenizer(code, return_tensors="pt", max_length=512, truncation=True)

# Generate the summary
summary_ids = model.generate(inputs['input_ids'], max_length=128, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the summary
print("Summary of the code in test.py:")
print(summary)
