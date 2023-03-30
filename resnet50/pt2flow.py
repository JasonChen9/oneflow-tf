import torch 
import oneflow as flow
import numpy as np
import cv2
from onnx2torch import convert
from torchvision.models import resnet50, ResNet50_Weights

onnx_model_path = "../model/pt2flow_resnet50.onnx"
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


def save_pt_model_as_onnx():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = preprocess_image(img)

    img = torch.from_numpy(img.copy())

    output = model(img)

    torch.onnx.export(model,               
                  img,                         
                  onnx_model_path,   
                  export_params=True,        
                  opset_version=13,          
                  do_constant_folding=True,  
                  input_names = ['input'],   
                  output_names = ['output'], 
            )
    with open('../model/imagenet-classes.txt') as f:
        CLASS_NAMES = f.readlines()
        print('PT Predicted:',
            CLASS_NAMES[np.argmax(output.detach().numpy()[0])])
    np.save("../model/pt2flow_resnet50_pt.npy", output.detach().numpy()[0])
    return output.detach().numpy()


def load_onnx_model_as_flow():
    torch_model = convert(onnx_model_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = preprocess_image(img)

    img = torch.from_numpy(img.copy())
    output = torch_model(img)

    with open('../model/imagenet-classes.txt') as f:
        CLASS_NAMES = f.readlines()
        print('OneFlow Predicted:',
            CLASS_NAMES[np.argmax(output.detach().numpy()[0])])
    np.save("../model/pt2flow_resnet50_flow.npy", output.detach().numpy()[0])
    return output.detach().numpy()

if __name__ == '__main__':
    if(isinstance(torch.zeros(2, 3), flow.Tensor)):
        load_onnx_model_as_flow()
    else:
        save_pt_model_as_onnx()


