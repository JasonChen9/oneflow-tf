import torch
import numpy as np
import oneflow as flow
from transformers import GPT2LMHeadModel

def save_pt_model_as_onnx():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    sample_input = torch.ones([1, 5], dtype=torch.int32)
    output = model(sample_input).logits.detach().numpy()
    
    if(isinstance(torch.zeros(2, 3), flow.Tensor)):
        path = "../model/pt2flow_gpt2_flow.npy"
    else:
        path = "../model/pt2flow_gpt2_pt.npy"
    np.save(path, output)
    return output

if __name__ == '__main__':
    save_pt_model_as_onnx()


