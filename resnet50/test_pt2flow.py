import os
import numpy as np

os.system("python3 pt2flow.py")
torch_output = np.load("../model/pt2flow_resnet50_pt.npy")

os.system("eval $(oneflow-mock-torch --lazy) && python3 pt2flow.py")
flow_output = np.load("../model/pt2flow_resnet50_flow.npy")

np.testing.assert_allclose(torch_output, flow_output, rtol=1e-05, atol=2e-05)
print("PASS")