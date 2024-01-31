import tensorflow as tf

model = tf.keras.models.load_model('models/gesture_inference_V2.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("models/model_V2.tflite", "wb") .write(tflite_model)
