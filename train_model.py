"""Train a classifier on Iris and save it to predictor/model.joblib."""
import joblib
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

joblib.dump(
    {"model": model, "classes": list(iris.target_names)},
    Path(__file__).parent / "predictor" / "model.joblib",
)
print("Model saved.")
