from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def complete_sentence(text):
    end_punctuations = {'.', '!', '?'}
    for i in range(len(text) - 1, -1, -1):
        if text[i] in end_punctuations:
            return text[:i+1]
    return text

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.bfloat16
)


def generate(input: str) -> str:
    input_ids = tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=20).input_ids
    
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=50,  # Adjust as needed
        num_beams=5,  # Number of beams for beam search
        early_stopping=True,  # Stop early if all beams finished
        no_repeat_ngram_size=2,  # Prevent repeating n-grams
        do_sample=True,  # Enable sampling
        temperature=0.7,  # Sampling temperature
        top_p=0.9,  # Nucleus sampling
    )

    # Decode and print the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text.replace(input, '').replace('\n', ' ').replace('\r', ' ')
    final_text = complete_sentence(generated_text)
    
    return final_text.strip()