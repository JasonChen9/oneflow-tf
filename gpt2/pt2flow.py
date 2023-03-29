import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
sample_output = model(
    torch.ones([1,5], dtype=torch.int32))
print("tf: ",sample_output[0])

