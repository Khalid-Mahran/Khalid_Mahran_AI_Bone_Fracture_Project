import os, json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

DATASET_DIR = "dataset"
MODEL_DIR = "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

os.makedirs(MODEL_DIR, exist_ok=True)

def load_paths():
    paths, labels = [], []
    for bone in sorted(os.listdir(DATASET_DIR)):
        bone_path = os.path.join(DATASET_DIR, bone)
        if not os.path.isdir(bone_path):
            continue
        for status in ["normal", "fractured"]:
            folder = os.path.join(bone_path, status)
            if not os.path.isdir(folder):
                continue
            label = f"{bone}_{status}"
            for img in os.listdir(folder):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(folder, img))
                    labels.append(label)
    return paths, labels

paths, labels = load_paths()

if len(paths) == 0:
    raise Exception("No images found. Put images inside dataset/bone/normal and dataset/bone/fractured")

class_names = sorted(list(set(labels)))
class_to_id = {c:i for i,c in enumerate(class_names)}

y = np.array([class_to_id[l] for l in labels])
x_train, x_val, y_train, y_val = train_test_split(
    paths, y, test_size=0.2, random_state=42, stratify=y
)

def preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess).shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

num_classes = len(class_names)

model = models.Sequential([
    layers.Input(shape=(224,224,3)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save("models/bone_model.keras")

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

print("DONE: model saved in models/bone_model.keras")
print("Classes:", class_names)
