import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import os

caffe.set_mode_cpu()

model_path = "../model/caffe2flow_resnet50.onnx"
img_path = "../img/cat.jpg"
MODEL_FILE = "/data/home/xiangguangyu/oneflow-tf/resnet50/caffe-onnx/caffemodel/resnet-50/resnet-50-model.prototxt"
PRETRAINED = "/data/home/xiangguangyu/oneflow-tf/resnet50/caffe-onnx/caffemodel/resnet-50/resnet-50-model.caffemodel"

def process_image(img_path, input_shape):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(input_shape)
    image = np.array(img, dtype=np.float32)
    image = image.transpose((2, 0, 1))[np.newaxis, ...]
    return image

def save_caffe_model_to_onnx():
    data_input = process_image(img_path, [224, 224])
    # load the model
    net = caffe.Net(MODEL_FILE,
                    PRETRAINED,
                    caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    out = net.forward_all(data=data_input)
    np.save('caffe_resnet50.npy', out['prob'])

    with open('../model/imagenet-classes.txt') as f:
        CLASS_NAMES = f.readlines()
        print('caffe Predicted:',
              CLASS_NAMES[out['prob'].argmax()])


    os.system("python caffe-onnx/convert2onnx.py " 
              "caffe-onnx/caffemodel/resnet-50/resnet-50-model.prototxt "  
              "caffe-onnx/caffemodel/resnet-50/resnet-50-model.caffemodel "  
              "caffe2flow_resnet50 ../model/")

if __name__ == '__main__':
    save_caffe_model_to_onnx()
