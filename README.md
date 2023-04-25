# oneflow-tf

## prepare env
```
//创建虚拟环境
conda create -n flow-pt-tf python=3.8

//激活环境
conda activate flow-pt-tf

//正确安装寒武纪版本oneflow。下述命令仅作为参考，具体安装方式请参考寒武纪版本oneflow相关文档
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117

//安装相关依赖库
while read requirement; do  pip install $requirement -i https://pypi.tuna.tsinghua.edu.cn/simple; done < requirements.txt
pip install git+https://github.com/Oneflow-Inc/oneflow_convert.git
pip install chardet 
pip install idna 


//oneflow 可以无缝使用pytorch的库, 所以可以通过onnx2torch来实现onnx模型转化为oneflow模型
pip install git+https://github.com/JasonChen9/onnx2torch.git
pip install git+https://github.com/JasonChen9/onnx2pytorch.git
```

## resnet50 
进入到resnet50目录中
```
cd resnet50
```
### oneflow convert to tensorflow
```
ONEFLOW_VM_MULTI_THREAD=0 python3 flow2tf.py
```

### tensorflow convert to oneflow
```
//开启mock
eval $(python3 -m oneflow.mock_torch --lazy)
ONEFLOW_VM_MULTI_THREAD=0 python3 tf2flow.py
```

### oneflow convert to pytorch
```
ONEFLOW_VM_MULTI_THREAD=0 python3 flow2pt.py
```

### pytorch convert to oneflow
```
ONEFLOW_VM_MULTI_THREAD=0 python3 test_pt2flow.py
``` 

## gpt2
进入到gpt2目录中
```
cd gpt2
```
### oneflow convert to tensorflow
使用libai运行gpt2并进行转化。libai的gpt2推理实现是在projects/MagicPrompt文件夹中，通过以下命令安装：
```
git clone --recursive https://github.com/Oneflow-Inc/oneflow-cambricon-models.git
cd oneflow-cambricon-models/libai
pip install pybind11
pip install -e .
```

libai的gpt2推理实现是在projects/MagicPrompt文件夹中，这个Magicprompt是我们自己用gpt2预训练后做推理的项目，用于将一个简单的句子转换成stable diffusion的咒语。接着把从 https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-model.zip 这里下载的模型解压到任意路径，并在 libai/ 下全局搜索/data/home/magicprompt将其替换为解压后的模型路径。

修改oneflow_onnx/oneflow2onnx/util.py的178行为`device_kwargs = dict(sbp=flow.sbp.broadcast, placement=flow.placement("mlu", ranks=[0]))`

修改projects/MagicPrompt/configs/gpt2_inference.py文件中67，68行以及projects/MagicPrompt/pipeline.py文件的99行为
之前模型解压路径。

最后把本项目的flow2tf.py文件内容覆盖到libai/onnx_export/gpt2_to_onnx.py中，修改文件第39行指定为之前模型解压路径下的model目录，第63和100行，指定为oneflow-tf/model目录下的onnx生成路径，例如path-
to/oneflow-tf/model/gpt2.onnx,然后就可以运行模型转换脚本。
```
python3 libai/onnx_export/gpt2_to_onnx.py 
```
即可执行转换和验证，之后回到本项目的gpt2目录中。
### tensorflow convert to oneflow
```
//开启mock
eval $(python3 -m oneflow.mock_torch --lazy)
ONEFLOW_VM_MULTI_THREAD=0 python3 tf2flow.py
```

### oneflow convert to pytorch
 flow2pt.py运行依赖于oneflow convert to tensorflow环节中的flow2tf.py生成的gpt2.onnx
```
ONEFLOW_VM_MULTI_THREAD=0 python3 flow2pt.py
```

### pytorch convert to oneflow
```
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
mv resnet-50-model.caffemodel caffe_model/resnet-50/
mv resnet-50-model.prototxt caffe_model/resnet-50/

//运行caffe2onnx.py 生成caffe的resnet-50 onnx模型
python caffe2onnx.py

//运行mxnet2onnx.py 生成mxnet的resnet-50 onnx模型
python mxnet2onnx.py


//回到oneflow环境，开启mock，从onnx导入到oneflow，并验证结果
conda activate flow-pt-tf
eval $(python3 -m oneflow.mock_torch --lazy)
ONEFLOW_VM_MULTI_THREAD=0 python3 onnx2flow.py
```