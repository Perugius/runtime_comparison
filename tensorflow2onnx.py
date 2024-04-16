import onnxruntime as ort
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx

input_signature = [tf.TensorSpec([None, 4, 60], tf.float64, name="input")]

# load model from .h5 file convert then save to onnx
model = tf.keras.saving.load_model("models/gesture_inference_V2.h5")
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=15)
onnx.save(onnx_model, "model_V2.onnx")