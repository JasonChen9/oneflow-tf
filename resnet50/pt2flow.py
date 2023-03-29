from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torch 
import os
import onnx
from onnx2pytorch import ConvertModel
import oneflow.mock_torch as mock

img = read_image("../img/cat.jpg")

def save_pt_model_as_onnx():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    batch = preprocess(img).unsqueeze(0)

    prediction = model(batch).squeeze(0).softmax(0)
    # prediction = model(batch)
    # torch.onnx.export(model, batch, '../model/pt2flow_resnet50')

    class_id = prediction.argmax().item()
    # score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(category_name)

def load_onnx_model_as_flow():
    # os.system("eval $(oneflow-mock-torch --lazy --verbose)")
    # onnx_model = onnx.load("../model/pt2flow_resnet50")
    # pytorch_model = ConvertModel(onnx_model)
    # output = pytorch_model(
    #     batch
    # )

    # print(output)

    # # np.testing.assert_allclose(output[0].detach().numpy(), onnx_pred[0], rtol=1e-2)
    print("pass")
    # os.system("eval $(oneflow-mock-torch disable)")

if __name__ == '__main__':
    save_pt_model_as_onnx()
    # with mock.enable():
    #     load_onnx_model_as_flow()

