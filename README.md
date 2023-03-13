# oneflow-tf

## flow convert to onnx
```
//创建虚拟环境
conda create -n flow python=3.8

//激活环境
conda activate flow

//安装oneflow 相关库
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117
conda install cudnn
pip install flowvision

//安装onnx相关库
pip install onnxruntime-gpu
pip install oneflow-onnx

//安装numpy
pip install mxnet-mkl==1.6.0 numpy==1.23.1

```

## onnx convert to flow
```
//创建虚拟环境
conda create -n tf python=3.8

//激活环境
conda activate tf

//安装tensorflow 以及相关库
conda install tensorflow-gpu
pip install tensorflow_probability==0.12.2 tensorflow-addons==0.14.0 keras opencv-python onnxruntime


```