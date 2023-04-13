import torch
import onnx
import numpy as np
import tensorflow as tf
from onnx2torch import convert
from onnx_tf.backend import prepare

def load_onnx_model_as_pt():
    torch_model = convert(
        "../model/gpt2.onnx")  # load onnx model
    output = torch_model(
        torch.ones([1,5], dtype=torch.int32)
    )
    return output

def load_onnx_model_as_tf():
    onnx_model = onnx.load(
        "../model/gpt2.onnx")  # load onnx model
    tf_model = prepare(onnx_model)
    input_ids = tf.ones(
        [1, 5], dtype=tf.int64
    )
    # warm up
    output = tf_model.run(inputs=input_ids)

    return output

if __name__ == "__main__":
    tf_out = load_onnx_model_as_tf()
    pt_out = load_onnx_model_as_pt()
    np.testing.assert_allclose(tf_out[0], pt_out, rtol=1e-2, atol=2e-5)
    print("PASS")

