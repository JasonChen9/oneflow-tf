import tensorflow as tf
import tf2onnx
import torch
import os
import onnxruntime as rt
import numpy as np
import onnx
from onnx2pytorch import ConvertModel
from transformers import TFGPT2LMHeadModel

onnx_model_path = "../model/tf2flow_gpt2.onnx"
def save_tf_model_as_onnx():
    model = TFGPT2LMHeadModel.from_pretrained("gpt2")

    sample_output = model(
        tf.ones([1,5], dtype=tf.int32))
    print("tf: ",sample_output[0])

    model.save_pretrained("../model/checkpoints/tfgpt2model", saved_model=True)

    command = "python3 -m tf2onnx.convert --saved-model ../model/checkpoints/tfgpt2model/saved_model/1 --opset 11  --output ../model/tf2flow_gpt2.onnx"

    os.system(command)

def load_onnx_to_flow_model():
    onnx_model = onnx.load(onnx_model_path)
    flow_model = ConvertModel(onnx_model).to('mlu')
    output = flow_model(
        attention_mask=torch.ones([1,5], dtype=torch.int32).to('mlu'),
        input_ids=torch.ones([1,5], dtype=torch.int32).to('mlu')
    )

    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(onnx_model_path, providers=providers)
    onnx_pred = m.run(['logits', 'past_key_values'], {"attention_mask": np.ones([1,5], dtype=np.int32),"input_ids":np.ones([1,5], dtype=np.int32)})
    
    np.testing.assert_allclose(output[0].detach().numpy(), onnx_pred[0], rtol=1e-2)
    print("pass")


if __name__ == '__main__':
    save_tf_model_as_onnx()
    load_onnx_to_flow_model()
