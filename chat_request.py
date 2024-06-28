# from transformers import AutoTokenizer, AutoModelForCausalLM
# from dotenv import load_dotenv
# import torch
# import os

# load_dotenv()
# hf_token = os.getenv("HF-KEY")

# def complete_sentence(text):
#     end_punctuations = {'.', '!', '?'}
#     for i in range(len(text) - 1, -1, -1):
#         if text[i] in end_punctuations:
#             return text[:i+1]
#     return text

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", use_auth_token=os.getenv("HF_API_TOKEN"))
# model = AutoModelForCausalLM.from_pretrained(
#     "google/gemma-2b-it",
#     torch_dtype=torch.bfloat16,
#     token=hf_token
# )

# def generate(input: str) -> str:
#     input_ids = tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=20).input_ids
    
#     outputs = model.generate(
#         input_ids=input_ids,
#         max_new_tokens=50, 
#         num_beams=5,  
#         early_stopping=True,  
#         no_repeat_ngram_size=2,  
#         do_sample=True,  
#         temperature=0.7,  
#         top_p=0.9,  
#     )

# #     # Decode and print the output
# #     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# #     generated_text = generated_text.replace(input, '').replace('\n', ' ').replace('\r', ' ')
# #     final_text = complete_sentence(generated_text)
    
#     return final_text.strip()
