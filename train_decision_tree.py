import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

DATASET_DIR = "dataset_binary"
REPORT_DIR = "reports"
IMG_SIZE = 64

os.makedirs(REPORT_DIR, exist_ok=True)

X = []
y = []

classes = ["fractured", "normal"]

for label, class_name in enumerate(classes):
    folder = os.path.join(DATASET_DIR, class_name)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            X.append(img.flatten())
            y.append(label)
        except:
            pass

X = np.array(X)
y = np.array(y)

print("Images loaded:", len(X))
print("Feature size:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\nDecision Tree Accuracy:", round(acc * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes))

with open("reports/decision_tree_report.txt", "w") as f:
    f.write(f"Decision Tree Accuracy: {round(acc * 100, 2)}%\n\n")
    f.write(classification_report(y_test, y_pred, target_names=classes))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot()
plt.title("Decision Tree Confusion Matrix")
plt.savefig("reports/decision_tree_confusion_matrix.png")

models = ["Custom CNN", "MobileNetV2", "Decision Tree"]
accuracies = [67.5, 82.0, round(acc * 100, 2)]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.savefig("reports/model_comparison.png")

print("\nSaved:")
print("- reports/decision_tree_report.txt")
print("- reports/decision_tree_confusion_matrix.png")
print("- reports/model_comparison.png")
