import oneflow as flow
from oneflow import nn
from flowvision.models import resnet50
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check
import cv2
import numpy as np
from onnx2torch import convert
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_path = "../model/flow2tf_resnet50.onnx"
img_path = "../img/cat.jpg"


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


def save_flow_model_as_onnx():
    class ResNet50Graph(nn.Graph):
        def __init__(self, eager_model):
            super().__init__()
            self.model = eager_model

        def build(self, x):
            return self.model(x)
        # 模型参数存储目录
    MODEL_PARAMS = '../model/checkpoints/flow2tf_resnet50'

    # 下载预训练模型并保存
    model = resnet50(pretrained=True)
    flow.save(model.state_dict(), MODEL_PARAMS, save_as_external_data=True)

    params = flow.load(MODEL_PARAMS)
    model = resnet50()
    model.load_state_dict(params)

    # 将模型设置为 eval 模式
    model.eval()

    resnet50_graph = ResNet50Graph(model)
    # 构建出静态图模型
    resnet50_graph._compile(flow.randn(1, 3, 224, 224))

    # 导出为 ONNX 模型并进行检查
    convert_to_onnx_and_check(resnet50_graph,
                              flow_weight_dir=MODEL_PARAMS,
                              onnx_model_path=model_path,
                              print_outlier=True,
                              dynamic_batch_size=True)


def load_onnx_model_as_pt():
    torch_model = convert(model_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = preprocess_image(img)

    img = torch.from_numpy(img.copy())
    out_torch = torch_model(img)

    with open('../model/imagenet-classes.txt') as f:
        CLASS_NAMES = f.readlines()
        print('OneFlow Predicted:',
              CLASS_NAMES[np.argmax(out_torch.detach().numpy()[0])])


if __name__ == '__main__':
    save_flow_model_as_onnx()
    load_onnx_model_as_pt()
