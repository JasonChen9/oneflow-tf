from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
import tf2onnx
import torch
from onnx2torch import convert
import os
text = "a dog"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def save_tf_model_as_onnx():
    model = TFGPT2LMHeadModel.from_pretrained(
        "gpt2", pad_token_id=tokenizer.eos_token_id
    )

    input = tokenizer.encode(text, return_tensors = 'tf')
    tf.random.set_seed(0)
    sample_output = model.generate(
        input, 
        do_sample=True, 
        max_length=97, 
        top_p=0.78, 
        top_k=0)
    print(sample_output[0])
    print(sample_output[0].shape)

    print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

    model.save_pretrained("../model/checkpoints/tfgpt2model", saved_model=True)

    command = "python -m tf2onnx.convert --saved-model ../model/checkpoints/tfgpt2model/saved_model/1 --opset 11  --output ../model/tf2flow_gpt2.onnx"

    os.system(command)

def load_onnx_to_flow_model():
    torch_model = convert("../model/tf2flow_gpt2.onnx")

    input = tokenizer.encode(text, return_tensors = 'pt')
    shape = input.shape

    output = torch_model(
        torch.ones(shape, dtype=torch.int32),input
    )
    print(output[0])
    print(output[0].shape)
    # print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == '__main__':
    save_tf_model_as_onnx()
    load_onnx_to_flow_model()