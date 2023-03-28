# oneflow-tf

## prepare env
```
//创建虚拟环境
conda create -n flow-pt-tf python=3.8

//激活环境
conda activate flow-pt-tf

//安装oneflow tensorflow相关库
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117
conda install --yes --file requirements.txt

//oneflow 可以无缝使用pytorch的库, 所以可以通过onnx2torch来实现onnx模型转化为oneflow模型
pip install git+https://github.com/JasonChen9/onnx2torch.git
pip install git+https://github.com/JasonChen9/onnx2pytorch.git
```

## resnet50 
### oneflow convert to tensorflow
```
python resnet50/flow2tf.py
```

### tensorflow convert to oneflow

```
//开启mock
eval $(oneflow-mock-torch --lazy)
python resnet50/tf2flow.py
```

## gpt2

### oneflow convert to tensorflow
```
//使用libai运行gpt2并进行转化 
//复制gpt2/flow2tf.py文件到libai/libai/onnx_export/gpt2_to_onnx.py  
//运行流程参照 https://github.com/Oneflow-Inc/OneTeam/issues/1946#issuecomment-1475897927
```

### tensorflow convert to oneflow

```
//开启mock
eval $(oneflow-mock-torch --lazy)
python gpt2/tf2flow.py
```


