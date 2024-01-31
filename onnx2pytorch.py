import onnx
from onnx2torch import convert
import torch

onnx_model = onnx.load("model.onnx_V2")
pytorch_model = convert(onnx_model)

torch.save(pytorch_model, "models/pymodel.pt")
