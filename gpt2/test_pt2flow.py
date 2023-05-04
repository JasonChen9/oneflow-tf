import os
import numpy as np

os.system("python3 pt2flow.py")
torch_output = np.load("../model/pt2flow_gpt2_pt.npy")

os.system("eval $(python3 -m oneflow.mock_torch --lazy) && python3 pt2flow.py")
flow_output = np.load("../model/pt2flow_gpt2_flow.npy")

np.testing.assert_allclose(torch_output, flow_output, rtol=1e-05, atol=1e-05)
print("精度验证通过")
