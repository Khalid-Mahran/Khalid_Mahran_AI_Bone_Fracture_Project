import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

MODEL_PATH = "models/bone_binary_model.keras"
CLASSES_PATH = "models/binary_class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)

selected_img = None

def estimate_recovery(status, age):
    if status == "normal":
        return "A", "No cast required", "No recovery time needed"

    if age < 18:
        return "C", "Cast required", "3 - 4 weeks"
    elif age <= 40:
        return "C", "Cast required", "4 - 6 weeks"
    elif age <= 60:
        return "D", "Cast required", "6 - 8 weeks"
    else:
        return "D", "Cast + doctor follow-up", "8 - 10 weeks"

def choose_image():
    global selected_img
    selected_img = filedialog.askopenfilename(
        title="Choose X-ray Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if selected_img:
        img = Image.open(selected_img)
        img.thumbnail((300, 300))
        tk_img = ImageTk.PhotoImage(img)
        image_label.config(image=tk_img)
        image_label.image = tk_img

def analyze():
    if not selected_img:
        messagebox.showerror("Error", "Please upload an X-ray image first.")
        return

    try:
        age = int(age_entry.get())
        height = float(height_entry.get())
        gender = gender_var.get()
    except:
        messagebox.showerror("Error", "Please enter valid age and height.")
        return

    img = image.load_img(selected_img, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0]
    idx = int(np.argmax(pred))
    status = class_names[idx]
    confidence = float(pred[idx]) * 100

    fracture_class, cast_decision, recovery_time = estimate_recovery(status, age)

    estimated_bone_length = round(height * 0.146, 1)
    fracture_location = round(estimated_bone_length * 0.65, 1)
    fracture_gap = 3

    report = f"""
AI Upper Limb X-ray Fracture Report

Patient Data:
Age: {age}
Height: {height} cm
Gender: {gender}

AI Result:
Condition: {status.upper()}
Confidence: {confidence:.2f}%

Estimated Bone Length: {estimated_bone_length} cm
Fracture Class: {fracture_class}
Cast Decision: {cast_decision}
Estimated Recovery Time: {recovery_time}
"""

    if status == "fractured":
        report += f"""
Estimated Fracture Location: {fracture_location} cm from the top
Estimated Fracture Gap: {fracture_gap} mm
"""

    report += """
Note:
This is an educational AI prototype and not a medical diagnosis.
"""

    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, report)

root = tk.Tk()
root.title("AI Upper Limb X-ray Fracture Analyzer")
root.geometry("800x700")

title = tk.Label(root, text="AI Upper Limb X-ray Fracture Analyzer", font=("Arial", 20, "bold"))
title.pack(pady=10)

form = tk.Frame(root)
form.pack(pady=10)

tk.Label(form, text="Age:").grid(row=0, column=0, padx=5, pady=5)
age_entry = tk.Entry(form)
age_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(form, text="Height (cm):").grid(row=0, column=2, padx=5, pady=5)
height_entry = tk.Entry(form)
height_entry.grid(row=0, column=3, padx=5, pady=5)

gender_var = tk.StringVar(value="male")
tk.Radiobutton(form, text="Male", variable=gender_var, value="male").grid(row=1, column=1)
tk.Radiobutton(form, text="Female", variable=gender_var, value="female").grid(row=1, column=2)

upload_btn = tk.Button(root, text="Upload X-ray Image", command=choose_image, width=25)
upload_btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

analyze_btn = tk.Button(root, text="Analyze", command=analyze, width=25, bg="green", fg="white")
analyze_btn.pack(pady=10)

result_text = tk.Text(root, height=18, width=90)
result_text.pack(pady=10)

root.mainloop()
