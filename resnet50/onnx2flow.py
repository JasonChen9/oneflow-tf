from onnx2pytorch import ConvertModel
import onnx
import onnxsim
import torch
import numpy as np
from PIL import Image
import os
import mxnet as mx
import numpy as np

caffe_model_path = "../model/caffe2flow_resnet50.onnx"
mxnet_model_path = "../model/mx2flow_resnet50.onnx"
img_path = "../img/cat.jpg"

def process_image(img_path, input_shape):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(input_shape)
    image = np.array(img, dtype=np.float32)
    image = image.transpose((2, 0, 1))[np.newaxis, ...]
    return image

def load_caffe_onnx_to_flow_model():
    new_onnx_model, _ = onnxsim.simplify(caffe_model_path)
    new_onnx_model_path = "../model/caffe2flow_resnet50-sim.onnx"
    onnx.save(new_onnx_model, new_onnx_model_path)
    onnx_model = onnx.load(new_onnx_model_path)
    torch_model = ConvertModel(onnx_model).to('mlu')
    data_input = process_image(img_path, [224, 224])
    torch_model.eval()
    img = torch.from_numpy(data_input.copy()).to('mlu')
    out_torch = torch_model(img)

    with open('../model/imagenet-classes.txt') as f:
        CLASS_NAMES = f.readlines()
        print('OneFlow Predicted:', CLASS_NAMES[np.argmax(out_torch.detach().numpy()[0])])

    b = np.load('caffe_resnet50.npy')
    np.testing.assert_allclose(out_torch.detach().numpy()[0], b[0], rtol=1e-02, atol=8e-03)
    print("CAFFE PASS")

def transform(image):
    resized = mx.image.resize_short(image, 224) #minimum 224x224 images
    cropped, crop_info = mx.image.center_crop(resized, (224, 224))
    normalized = mx.image.color_normalize(cropped.astype(np.float32)/255,
                                      mean=mx.nd.array([0.485, 0.456, 0.406]),
                                      std=mx.nd.array([0.229, 0.224, 0.225]))
    # the network expect batches of the form (N,3,224,224)
    transposed = normalized.transpose((2,0,1))  # Transposing from (224, 224, 3) to (3, 224, 224)
    batchified = transposed.expand_dims(axis=0) # change the shape from (3, 224, 224) to (1, 3, 224, 224)
    return batchified

def load_mxnet_onnx_to_flow_model():
    new_onnx_model, _ = onnxsim.simplify(mxnet_model_path)
    new_onnx_model_path = "../model/mx2flow_resnet50-sim.onnx"
    onnx.save(new_onnx_model, new_onnx_model_path)
    onnx_model = onnx.load(new_onnx_model_path)
    torch_model = ConvertModel(onnx_model).eval().to('mlu')

    image = mx.image.imread(img_path)
    data_input = transform(image).asnumpy()
    img = torch.from_numpy(data_input.copy()).to('mlu')
    out_torch = torch_model(img)

    with open('../model/imagenet-classes.txt') as f:
        CLASS_NAMES = f.readlines()
        print('OneFlow Predicted:', CLASS_NAMES[np.argmax(out_torch.detach().numpy()[0])])

    b = np.load('mxnet_resnet50.npy')
    np.testing.assert_allclose(out_torch.detach().numpy()[0], b, rtol=1e-05, atol=2e-05)
    print("MXNET PASS")

if __name__ == '__main__':
    load_caffe_onnx_to_flow_model()
    load_mxnet_onnx_to_flow_model()