import onnxruntime as ort
import numpy as np
import keras
import tensorflow as tf
import tf2onnx
import time
import torch


# np array for onnx and tf, torch tensor for pytorch
input1 = np.load("test_data/x_train.npy")
print(input1.shape)
input_pt = torch.from_numpy(input1)
input_tf_lite = np.reshape(np.float32(np.load("test_data/rand_model_13.npy")), (1, 15008, 2))
#input_tf_lite = np.reshape(np.float32(input1[25]), (1, 4, 60))
print(input_tf_lite.shape)

# pytorch
pytorch_latency_list = []

model_pt = torch.load("models/pymodel.pt")
device = torch.device("cpu")
model_pt.to(device)
model_pt.eval()

for i in range(0, 1000):
    start_pt = time.time()
    output_pt = model_pt(input_pt)
    end_pt = time.time()
    time_elapsed_pt = end_pt - start_pt
    pytorch_latency_list.append(time_elapsed_pt)
pytorch_latency_array = np.array(pytorch_latency_list)
time_elapsed_pt_avg = np.mean(pytorch_latency_array)

# tf
tf_latency_list = []
model_tf = tf.keras.models.load_model("models/gesture_inference_V2.h5")

with tf.device('/cpu:0'):
    for i in range(0, 1000):
        start_tf = time.time()
        # Assuming input1 is a properly formatted tensor or numpy array
        results_tf = model_tf(input1)
        end_tf = time.time()
        time_elapsed_tf = end_tf - start_tf
        tf_latency_list.append(time_elapsed_tf)

tf_latency_array = np.array(tf_latency_list)
time_elapsed_tf_avg = np.mean(tf_latency_array)

# # tf-lite only invoke 1 output from 1 input
# interpreter = tf.lite.Interpreter(model_path="models/model_13_egc.tflite")
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# input_shape = input_details[0]['shape']
# interpreter.set_tensor(input_details[0]['index'], input_tf_lite)
# while True:
#     start_tflite = time.time()
#     interpreter.invoke()
#     end_tflite = time.time()
#     time_elapsed_tflite = end_tflite - start_tflite
#     print("tflite rt: "+str(time_elapsed_tflite))
# output_tflite = interpreter.get_tensor(output_details[0]['index'])
# print(output_tflite)

# onnx

onnx_latency_list = []

for i in range(0, 1000):
    sess = ort.InferenceSession("models/model_V2.onnx", providers=["CPUExecutionProvider"])
    start_onnx = time.time()
    results_ort = sess.run(None, {"input": input1})
    end_onnx = time.time()
    time_elapsed_onnx = end_onnx - start_onnx
    onnx_latency_list.append(time_elapsed_onnx)
onnx_latency_array = np.array(onnx_latency_list)
time_elapsed_onnx_avg = np.mean(onnx_latency_array)



print("tensorflow rt: "+str(time_elapsed_tf_avg))
print("onnx rt: "+str(time_elapsed_onnx))
print("pytorch rt: "+str(time_elapsed_pt_avg))
#print("tflite rt: "+str(time_elapsed_tflite))