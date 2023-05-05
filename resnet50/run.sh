#!/bin.bash
set -e

## resnet50 
cd resnet50
### tensorflow convert to oneflow
eval $(python3 -m oneflow.mock_torch --lazy)
ONEFLOW_VM_MULTI_THREAD=0 python3 tf2flow.py
eval $(python3 -m oneflow.mock_torch disable)
### pytorch convert to oneflow
ONEFLOW_VM_MULTI_THREAD=0 python3 test_pt2flow.py

## gpt2
cd ../gpt2
### tensorflow convert to oneflow
eval $(python3 -m oneflow.mock_torch --lazy)
ONEFLOW_VM_MULTI_THREAD=0 python3 tf2flow.py
eval $(python3 -m oneflow.mock_torch disable)
### pytorch convert to oneflow
ONEFLOW_VM_MULTI_THREAD=0 python3 test_pt2flow.py