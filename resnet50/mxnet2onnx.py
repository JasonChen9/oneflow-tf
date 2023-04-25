import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import numpy as np

onnx_model_path = "../model/mx2flow_resnet50.onnx"
img_path = "../img/cat.jpg"

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


def predict(model, image, k):
    predictions = model(transform(image))
    np.save('mxnet_resnet50.npy', predictions[0].asnumpy())
    predictions.softmax()
    top_pred = predictions.topk(k=k)[0].asnumpy()
    for index in top_pred:
        with open('../model/imagenet-classes.txt') as f:
            CLASS_NAMES = f.readlines()
            probability = predictions[0][int(index)]
            category = CLASS_NAMES[int(index)]
            print("{}: {:.2f}%".format(category, probability.asscalar()*100))
        print('')

def save_caffe_model_to_onnx():
    ctx = mx.cpu()
    resnet50 = vision.resnet50_v1(pretrained=True, ctx=ctx)
    model_name = 'resnet50_v1'

    filename = img_path
    image = mx.image.imread(filename)

    resnet50.hybridize()

    predict(resnet50, image, 3)

    resnet50.export(model_name)
    from mxnet.contrib import onnx as onnx_mxnet
    params = './'+model_name+'-0000.params'
    sym='./'+model_name+'-symbol.json'
    in_shapes = [(1, 3, 224, 224)]
    onnx_mxnet.export_model(sym, params, in_shapes, np.float32, onnx_model_path)
    print('onnx export done')


if __name__ == '__main__':
    save_caffe_model_to_onnx()
