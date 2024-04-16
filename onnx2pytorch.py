import onnx
from onnx2torch import convert
import torch

onnx_model = onnx.load("model_V2.onnx")
pytorch_model = convert(onnx_model)

torch.save(pytorch_model, "models/pymodel.pt")
