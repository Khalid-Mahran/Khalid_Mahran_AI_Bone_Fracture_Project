import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "outputs/extracted_features.csv"

df = pd.read_csv(DATA_PATH)

feature_columns = [
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "contrast",
    "edge_density",
    "laplacian_variance",
    "width",
    "height",
    "aspect_ratio",
]

X = df[feature_columns]
y = df["label"]

if len(set(y)) < 2:
    raise SystemExit("Dataset must contain both fractured and not_fractured/normal images.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    ),
    "Artificial Neural Network": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=800,
            random_state=42
        ))
    ])
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    results.append({
        "model": name,
        "accuracy": round(acc * 100, 2),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features_used": len(feature_columns)
    })

    safe_name = name.lower().replace(" ", "_")
    joblib.dump(model, f"models/{safe_name}.joblib")

    print("\n==============================")
    print(name)
    print("==============================")
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("Classification Report:")
    print(classification_report(
        y_test,
        preds,
        target_names=["not_fractured", "fractured"],
        zero_division=0
    ))

results_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False)
results_df.to_csv("outputs/model_comparison.csv", index=False)

print("\nFinal Model Comparison:")
print(results_df.to_string(index=False))
print("\nSaved: outputs/model_comparison.csv")
print("Saved models inside: models/")
