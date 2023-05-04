import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import tf2onnx
import onnxruntime as rt
import torch
from onnx2torch import convert
import onnx
import onnxsim
import time

img_path = "../img/cat.jpg"
onnx_model_path = "../model/tf2flow_resnet50.onnx"
def save_tf_model_as_onnx():
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    model = ResNet50(weights='imagenet')

    preds = model.predict(x)
    np.save('tf_resnet50.npy', preds[0])
    print('Keras Predicted:', decode_predictions(preds, top=3)[0])

    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=13, output_path=onnx_model_path)
    output_names = [n.name for n in model_proto.graph.output]

    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(onnx_model_path, providers=providers)
    onnx_pred = m.run(output_names, {"input": x})

    print('ONNX Predicted:', decode_predictions(onnx_pred[0], top=3)[0])

    np.testing.assert_allclose(preds, onnx_pred[0], rtol=1e-3)

def load_onnx_to_flow_model():
    new_onnx_model, _ = onnxsim.simplify(onnx_model_path)
    new_onnx_model_path = "../model/tf2flow_resnet50-sim.onnx"
    onnx.save(new_onnx_model, new_onnx_model_path)
    torch_model = convert(new_onnx_model_path).to('mlu')
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img = torch.from_numpy(x.copy()).to('mlu')
    print("运行中... 请通过 cnmon 命令查看显存占用")
    start_time = time.time()
    for i in range(10):
        out_torch = torch_model(img)
    end_time = time.time()
    print("运行结束,共运行 10 次，平均速度为 {:.2f}ms".format((end_time - start_time)* 100))

    with open('../model/imagenet-classes.txt') as f:
        CLASS_NAMES = f.readlines()
        print('OneFlow Predicted:', CLASS_NAMES[np.argmax(out_torch.detach().numpy()[0])])
        b = np.load('tf_resnet50.npy')
        np.testing.assert_allclose(out_torch.detach().numpy()[0], b, rtol=1e-05, atol=2e-05)

    print("精度验证通过")

if __name__ == '__main__':
    save_tf_model_as_onnx()
    load_onnx_to_flow_model()
