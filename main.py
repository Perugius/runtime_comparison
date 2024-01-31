import onnxruntime as ort
import numpy as np
import keras
import tensorflow as tf
import tf2onnx
import time
import torch


# np array for onnx and tf, torch tensor for pytorch
input1 = np.load("test_data/x_test.npy")
input_pt = torch.from_numpy(input1)
input_tf_lite = np.reshape(np.float32(np.load("test_data/rand_model_13.npy")), (1, 15008, 2))
#input_tf_lite = np.reshape(np.float32(input1[25]), (1, 4, 60))
print(input_tf_lite.shape)

# pytorch
model_pt = torch.load("models/pymodel.pt")
model_pt.eval()
start_pt = time.time()
output_pt = model_pt(input_pt)
end_pt = time.time()
time_elapsed_pt = end_pt - start_pt

# tf
model_tf = tf.keras.saving.load_model("models/gesture_inference_V2.h5")
start_tf = time.time()
results_tf = model_tf(input1)
end_tf = time.time()
time_elapsed_tf = end_tf - start_tf

# tf-lite only invoke 1 output from 1 input
interpreter = tf.lite.Interpreter(model_path="models/model_13_egc.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], input_tf_lite)
while True:
    start_tflite = time.time()
    interpreter.invoke()
    end_tflite = time.time()
    time_elapsed_tflite = end_tflite - start_tflite
    print("tflite rt: "+str(time_elapsed_tflite))
output_tflite = interpreter.get_tensor(output_details[0]['index'])
print(output_tflite)

# onnx
sess = ort.InferenceSession("model.onnx_V2", providers=["CPUExecutionProvider"])
start_onnx = time.time()
results_ort = sess.run(None, {"input": input1})
end_onnx = time.time()
time_elapsed_onnx = end_onnx - start_onnx



print("tensorflow rt: "+str(time_elapsed_tf))
print("onnx rt: "+str(time_elapsed_onnx))
print("pytorch rt: "+str(time_elapsed_pt))
print("tflite rt: "+str(time_elapsed_tflite))