from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torch 
import os
import onnx
import oneflow.mock_torch as mock
import numpy as np
import cv2
from onnx2torch import convert

img_path = "../img/cat.jpg"
onnx_model_path = "../model/pt2flow_resnet50.onnx"

def preprocess_image(img, input_hw=(224, 224)):
    h, w, _ = img.shape

    # 使用图像的较长边确定缩放系数
    is_wider = True if h <= w else False
    scale = input_hw[1] / w if is_wider else input_hw[0] / h

    # 对图像进行等比例缩放
    processed_img = cv2.resize(
        img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # 归一化
    processed_img = np.array(processed_img, dtype=np.float32) / 255

    # 将图像填充到 ONNX 模型预设尺寸
    temp_img = np.zeros((input_hw[0], input_hw[1], 3), dtype=np.float32)
    temp_img[:processed_img.shape[0],
             :processed_img.shape[1], :] = processed_img
    processed_img = temp_img

    # 调整轴的顺序并在最前面添加 batch 轴
    processed_img = np.expand_dims(processed_img.transpose(2, 0, 1), axis=0)

    return processed_img


def save_pt_model_as_onnx():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # preprocess = weights.transforms()
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = preprocess_image(img)

    img = torch.from_numpy(img.copy())

    output = model(img)

    torch.onnx.export(model,               # model being run
                  img,                         # model input (or a tuple for multiple inputs)
                  onnx_model_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
            )
    with open('../model/imagenet-classes.txt') as f:
        CLASS_NAMES = f.readlines()
        print('PT Predicted:',
            CLASS_NAMES[np.argmax(output.detach().numpy()[0])])
    return output.detach().numpy()


def load_onnx_model_as_flow():
    # with mock.enable():
    #     import torch
    torch_model = convert(onnx_model_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = preprocess_image(img)

    img = torch.from_numpy(img.copy())
    out_torch = torch_model(img)

    with open('../model/imagenet-classes.txt') as f:
        CLASS_NAMES = f.readlines()
        print('OneFlow Predicted:',
            CLASS_NAMES[np.argmax(out_torch.detach().numpy()[0])])
    return out_torch.detach().numpy()

if __name__ == '__main__':
    pt_out = save_pt_model_as_onnx()
    flow_out = load_onnx_model_as_flow()
    np.testing.assert_allclose(pt_out, flow_out, rtol=1e-2, atol=2e-5)
    print("PASS")


