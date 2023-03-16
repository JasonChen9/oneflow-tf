# oneflow-tf

## flow convert to tf
```
//创建虚拟环境
conda create -n flow-tf python=3.8

//激活环境
conda activate flow-tf

//安装oneflow 相关库
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117
conda install cudnn
pip install flowvision
pip install oneflow-onnx
pip install mxnet-mkl==1.6.0 numpy==1.23.1
pip install tensorflow-cpu
pip install tensorflow_probability==0.12.2 tensorflow-addons==0.14.0 keras opencv-python onnxruntime

python flow2tf.py
```


## tf convert to torch

```
python tf2flow.py

```