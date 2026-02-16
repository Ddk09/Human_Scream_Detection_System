import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_csv("features_augmented.csv")

data['label'] = pd.to_numeric(data['label'], errors='coerce')

data = data.dropna()

X = data.drop("label", axis=1)
y = data["label"].astype(int)

print("Total rows after cleaning:", len(data))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(rf, "scream_rf_model.pkl")
print("\n scream_rf_model.pkl")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Scream", "Scream"],
            yticklabels=["Non-Scream", "Scream"])
plt.title("Confusion Matrix of Random Forest Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices], color="orange", align="center")
plt.xticks(range(len(importances)), X.columns[indices], rotation=90)
plt.title("Feature Importance (MFCC + ZCR + Centroid + Rolloff)")
plt.tight_layout()
plt.show()

report = classification_report(y_test, y_pred, output_dict=True)
metrics = pd.DataFrame(report).transpose()
metrics = metrics.drop("accuracy", errors="ignore").drop("support", axis=1, errors="ignore")

plt.figure(figsize=(7, 5))
metrics.iloc[:2, :3].plot(kind="bar", color=["skyblue", "lightgreen", "salmon"])
plt.title("Precision, Recall, and F1-score per Class")
plt.xlabel("Class")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(ticks=[0, 1], labels=["Non-Scream", "Scream"], rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()

print("\nNumber of features (columns):", data.shape[1] - 1)
print("Feature names:", list(data.columns[:-1]))






















