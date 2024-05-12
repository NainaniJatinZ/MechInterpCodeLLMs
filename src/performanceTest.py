from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Evaluation function
def evaluate_next_token_accuracy(prompts_dataset, tokenizer, model):
    total_examples = len(prompts_dataset)
    correct_predictions = 0
    
    for input_text, next_token in prompts_dataset:
        # Tokenize input text
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate predictions
        with torch.no_grad():
            logits = model(input_ids)[0][:, -1, :]  # Get logits for the next token
            predicted_token_id = torch.argmax(logits, dim=-1).item()
            predicted_token = tokenizer.decode(predicted_token_id)
        
        # Check if prediction matches the ground truth
        if predicted_token == next_token:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_examples
    return accuracy

# Example prompts dataset
prompts = [("The cat", "is"), ("I am", "going"), ("OpenAI is a", "research")]

# Evaluate accuracy
accuracy = evaluate_next_token_accuracy(prompts, tokenizer, model)
print("Accuracy:", accuracy)