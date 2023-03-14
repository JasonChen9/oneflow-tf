import torch
import numpy as np
from onnx2torch import convert
import onnxruntime as ort
import cv2


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
    processed_img = np.expand_dims(processed_img, axis=0)

    return processed_img


if __name__ == '__main__':
    # Path to ONNX model
    onnx_model_path = '../tf2onnx/resnet50.onnx'
    img_path = "../img/cat.jpg"
    # You can pass the path to the onnx model to convert it or...
    torch_model_1 = convert(onnx_model_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = preprocess_image(img)
    # img = np.transpose(img,(0,2,3,1))
    img = torch.from_numpy(img)
    out_torch = torch_model_1(img)
    # ort_sess = ort.InferenceSession(onnx_model_path)
    # outputs_ort = ort_sess.run(None, [{'input': img} ])
    with open('../onnx2tf/imagenet-classes.txt') as f:
        CLASS_NAMES = f.readlines()
        print(CLASS_NAMES[np.argmax(out_torch.detach().numpy()[0])])

    # # Check the Onnx output against PyTorch
    # print(torch.max(torch.abs(outputs_ort - out_torch.detach().numpy())))
    # print(np.allclose(outputs_ort, out_torch.detach().numpy(), atol=1.e-7))