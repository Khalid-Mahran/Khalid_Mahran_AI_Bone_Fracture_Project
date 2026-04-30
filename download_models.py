import os
import requests

MODEL_URL = "PUT_YOUR_DIRECT_LINK_HERE"  # حط لينك الموديل
SAVE_PATH = "models/bone_binary_model.keras"

os.makedirs("models", exist_ok=True)

if not os.path.exists(SAVE_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(SAVE_PATH, "wb") as f:
        f.write(r.content)
    print("Done.")
else:
    print("Model already exists.")
