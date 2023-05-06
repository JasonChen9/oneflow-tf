# oneflow-tf

## prepare env
```
//创建虚拟环境
conda create -n flow-pt-tf python=3.8

//激活环境
conda activate flow-pt-tf

//正确安装寒武纪版本oneflow。
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/dev_cambricon_mlu/mlu

//安装相关依赖库
while read requirement; do  pip install $requirement -i https://pypi.tuna.tsinghua.edu.cn/simple; done < requirements.txt
pip install git+https://github.com/Oneflow-Inc/oneflow_convert.git

//oneflow 可以无缝使用pytorch的库, 所以可以通过onnx2torch来实现onnx模型转化为oneflow模型
pip install git+https://github.com/JasonChen9/onnx2torch.git
pip install git+https://github.com/JasonChen9/onnx2pytorch.git
```

## resnet50 
进入到resnet50目录中
```
cd resnet50
```

### tensorflow convert to oneflow
```
//开启mock
eval $(python3 -m oneflow.mock_torch --lazy)
ONEFLOW_VM_MULTI_THREAD=0 python3 tf2flow.py
```

### pytorch convert to oneflow
```
eval $(python3 -m oneflow.mock_torch disable)
ONEFLOW_VM_MULTI_THREAD=0 python3 test_pt2flow.py
``` 

### caffe & Mxnet convert to oneflow
```
//安装依赖环境和依赖库,注意由于需要安装caffe，因此需要一个全新的python3.7环境
conda create -n caffe-mxnet python=3.7
conda activate caffe-mxnet
conda install caffe
pip install mxnet
pip install protobuf==3.19.6
pip install onnx==1.10.2
pip install onnxruntime

//导入依赖库
git clone https://github.com/JasonChen9/caffe-onnx.git
cd caffe-onnx

//下载caffe的resnet50模型到caffe-onnx下的caffe_model/resnet-50路径下
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/resnet50-caffe/resnet-50-model.caffemodel
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/resnet50-caffe/resnet-50-model.prototxt
mv resnet-50-model.caffemodel caffemodel/resnet-50/
mv resnet-50-model.prototxt caffemodel/resnet-50/

//回到resnet50目录
cd ..

//运行caffe2onnx.py 生成caffe的resnet-50 onnx模型
ONEFLOW_VM_MULTI_THREAD=0 python3 caffe2onnx.py

//运行mxnet2onnx.py 生成mxnet的resnet-50 onnx模型
ONEFLOW_VM_MULTI_THREAD=0 python3 mxnet2onnx.py


//回到oneflow环境，开启mock，从onnx导入到oneflow，并验证结果
conda activate flow-pt-tf
eval $(python3 -m oneflow.mock_torch --lazy)
ONEFLOW_VM_MULTI_THREAD=0 python3 onnx2flow.py
```

## gpt2
进入到gpt2目录中
```
cd gpt2
```

### tensorflow convert to oneflow
```
//开启mock
eval $(python3 -m oneflow.mock_torch --lazy)
ONEFLOW_VM_MULTI_THREAD=0 python3 tf2flow.py
```

### pytorch convert to oneflow
```
eval $(python3 -m oneflow.mock_torch disable)
ONEFLOW_VM_MULTI_THREAD=0 python3 test_pt2flow.py
``` 
