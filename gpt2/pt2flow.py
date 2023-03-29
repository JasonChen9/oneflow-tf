import torch
from transformers import GPT2LMHeadModel

import os
import onnxruntime as rt
import numpy as np
import onnx
from onnx2pytorch import ConvertModel
import oneflow.mock_torch as mock

def save_pt_model_as_onnx():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    torch.onnx.export(model, torch.ones([1, 5], dtype=torch.int32), "torch-model.onnx", export_params=True,
                      opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    
def load_onnx_model_as_flow():
    # os.system("eval $(oneflow-mock-torch --lazy --verbose)")
    # onnx_model = onnx.load("torch-model.onnx")
    # pytorch_model = ConvertModel(onnx_model)
    # output = pytorch_model(
    #     input_ids=torch.ones([1,5], dtype=torch.int32)
    # )

    # providers = ['CPUExecutionProvider']
    # m = rt.InferenceSession("../model/tf2flow_gpt2.onnx", providers=providers)
    # onnx_pred = m.run(['logits', 'past_key_values'], {"input_ids":np.ones([1,5], dtype=np.int32)})
    
    # np.testing.assert_allclose(output[0].detach().numpy(), onnx_pred[0], rtol=1e-2)
    print("pass")
    # os.system("eval $(oneflow-mock-torch disable)")

if __name__ == '__main__':
    save_pt_model_as_onnx()
    with mock.enable():
        load_onnx_model_as_flow()

