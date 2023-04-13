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
使用libai运行gpt2并进行转化。libai的gpt2推理实现是在projects/MagicPrompt文件夹中，通过以下命令安装libai：
```
git clone git@github.com:Oneflow-Inc/libai.git
pip install pybind11
cd libai
pip install -e .
```
接着，新建一个文件夹，并把 https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion/tree/main 下的 `config.json`、`merges.txt`、`pytorch_model.bin`、`vocab.json`这四个文件下载到这个文件夹中，假设这个文件夹路径为/home/xiangguangyu/onefow_gpt2_model。接着，我们修改 libai/projects/MagicPrompt/configs/gpt2_inference.py 中 /data/home/magicprompt 改为 /home/xiangguangyu/onefow_gpt2_model，并修改66，67行的
```
vocab_file="/data/home/magicprompt/vocab.json", 
merges_file="/data/home/magicprompt/merges.txt",
```
改为刚才保存的模型
```
vocab_file="/data/home/xiangguangyu/oneflow_gpt2_model/vocab.json", 
merges_file="/data/home/xiangguangyu/oneflow_gpt2_model/merges.txt"
```
同时修改 libai/projects/MagicPrompt/pipeline.py 中的第100行为model_path="/home/xiangguangyu/onefow_gpt2_model"，并且把101行改成mode="libai"。
最后把本项目的flow2tf.py文件内容覆盖到libai/libai/onnx_export/gpt2_to_onnx.py中，修改文件第39行指定为刚才创建的文件夹下model目录，并在刚才创建的文件夹下新建model，第63和100行，指定为本项目下model目录，例如/ssd/dataset/xgy/oneflow-tf/model/gpt2.onnx,运行libai/libai/onnx_export/gpt2_to_onnx.py即可执行转换。

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
