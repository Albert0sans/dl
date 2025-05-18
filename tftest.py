import tensorflow as tf

converter = tf.lite.TFLiteConvertepir.from_saved_model("cur")  # Folder path
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as model.tflite")