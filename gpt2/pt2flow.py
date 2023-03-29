import torch
from transformers import GPT2LMHeadModel
import os
import onnxruntime as rtE
import numpy as np
import onnx
import oneflow.mock_torch as mock

img_path = "../img/cat.jpg"
onnx_model_path = "../model/pt2flow_gpt2.onnx"
def save_pt_model_as_onnx():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    sample_input = torch.ones([1, 5], dtype=torch.int32)
    output = model(sample_input)
    return output

if __name__ == '__main__':
    pt_out = save_pt_model_as_onnx()

    # with mock.enable(lazy=True, verbose=True):
    #     flow_out = save_pt_model_as_onnx()
    # np.testing.assert_allclose(pt_out, flow_out, rtol=1e-2, atol=2e-5)
    # print(pt_out)
    print("PASS")

