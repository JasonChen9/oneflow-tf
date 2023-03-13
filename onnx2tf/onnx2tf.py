import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import cv2
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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


if __name__ == '__main__':
    with tf.device('/GPU:0'):
        with open('imagenet-classes.txt') as f:
            img_path = '../img/cat.jpg'

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = preprocess_image(img)

            # load model
            time_start = time.time()
            onnx_model = onnx.load(
                "../flow2onnx/model.onnx")  # load onnx model
            time_end = time.time()
            print('prepare time cost', (time_end-time_start), 's')

            # prepare model
            time_start = time.time()
            model = prepare(onnx_model)
            time_end = time.time()
            print('prepare time cost', (time_end-time_start), 's')

            # warm up
            time_start = time.time()
            out = model.run(inputs=img, shape=(1, 3, 224, 224))
            time_end = time.time()
            print('warmup time cost', (time_end-time_start), 's')

            # run the loaded model
            time_start = time.time()
            for _ in range(10):
                out = model.run(inputs=img, shape=(1, 3, 224, 224))
            time_end = time.time()
            print('model time cost', (time_end-time_start)/10., 's')

            # out label
            CLASS_NAMES = f.readlines()
            print(CLASS_NAMES[np.argmax(out[0])])
