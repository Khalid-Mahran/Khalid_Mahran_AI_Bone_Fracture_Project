import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("models/bone_binary_model.keras")

with open("models/binary_class_names.json", "r") as f:
    class_names = json.load(f)

img_path = input("Enter image path: ")

img = image.load_img(img_path, target_size=(224,224))
arr = image.img_to_array(img) / 255.0
arr = np.expand_dims(arr, axis=0)

pred = model.predict(arr)[0]
idx = int(np.argmax(pred))

print("\n===== Prediction Result =====")
print("Prediction:", class_names[idx])
print("Confidence:", round(float(pred[idx]) * 100, 2), "%")
