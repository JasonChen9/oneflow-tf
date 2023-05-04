import torch
import numpy as np
import oneflow as flow
from transformers import GPT2LMHeadModel
import time

def save_pt_model_as_onnx():
    if(isinstance(torch.zeros(2, 3), flow.Tensor)):
        path = "../model/pt2flow_gpt2_flow.npy"
        device = 'mlu'
    else:
        path = "../model/pt2flow_gpt2_pt.npy"
        device = 'cpu'
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    sample_input = torch.ones([1, 5], dtype=torch.int32).to(device)
    if(isinstance(torch.zeros(2, 3), flow.Tensor)): print("运行中... 请通过 cnmon 命令查看显存占用")
    start_time = time.time()
    for i in range(10):
        output = model(sample_input).logits.detach().numpy()
    end_time = time.time()
    if(isinstance(torch.zeros(2, 3), flow.Tensor)): print("运行结束,共运行 10 次，平均速度为 {:.2f}ms".format((end_time - start_time)* 100))
    np.save(path, output)
    return output

if __name__ == '__main__':
    save_pt_model_as_onnx()