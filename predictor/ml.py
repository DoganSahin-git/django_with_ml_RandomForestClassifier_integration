"""Loads the trained model once per process and exposes a predict() helper."""
from functools import lru_cache
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).parent / "model.joblib"


@lru_cache(maxsize=1)
def _load():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train_model.py` first.")
    return joblib.load(MODEL_PATH)


def predict(features):
    """features: [sepal_length, sepal_width, petal_length, petal_width] -> dict"""
    bundle = _load()
    model, classes = bundle["model"], bundle["classes"]
    probs = model.predict_proba([features])[0]
    best = int(probs.argmax())
    return {
        "species": classes[best],
        "confidence": round(float(probs[best]), 3),
        "probabilities": {c: round(float(p), 3) for c, p in zip(classes, probs)},
    }
