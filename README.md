# oneflow-tf

## prepare env
```
//创建虚拟环境
conda create -n flow-pt-tf python=3.8

//激活环境
conda activate flow-pt-tf

//安装oneflow 相关库
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117
conda install cudnn
pip install flowvision
pip install oneflow-onnx
pip install mxnet-mkl==1.6.0 numpy==1.23.1
pip install tensorflow-cpu
pip install tensorflow_probability==0.12.2 tensorflow-addons==0.14.0 keras opencv-python onnxruntime

//安装转化需要的库
pip install oneflow_onnx
pip install onnx_tf
pip install -U tf2onnx
pip install onnx2torch
```

## flow convert to tf
```
python flow2tf.py
```


## tf convert to torch

```
python tf2flow.py

```