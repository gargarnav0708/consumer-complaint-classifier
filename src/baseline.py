"""Baseline model: frozen DistilBERT features + Logistic Regression.

DistilBERT is NOT trained here. We only use it to turn each complaint into a
768-number vector (the [CLS] token embedding), then fit a simple classifier.

Run:  python -m src.baseline
"""
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from src.config import BASE_MODEL, MAX_LENGTH, MODEL_DIR, PRODUCTS
from src.data import prepare_data
from src.evaluate import report


def embed(texts, tokenizer, model, device, batch_size=32):
    """Return one [CLS] embedding per text, shape (n_texts, 768)."""
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start:start + batch_size],
            padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state  # (batch, tokens, 768)
        vectors.append(hidden[:, 0].cpu().numpy())     # first token = [CLS]
    return np.vstack(vectors)


def main():
    train_df, test_df = prepare_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModel.from_pretrained(BASE_MODEL).to(device).eval()

    print(f"Embedding complaints on {device} (slow on CPU)...")
    X_train = embed(train_df["text"].tolist(), tokenizer, model, device)
    X_test = embed(test_df["text"].tolist(), tokenizer, model, device)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, train_df["product"])

    print(f"Train accuracy: {clf.score(X_train, train_df['product']):.3f}")
    print(f"Test accuracy:  {clf.score(X_test, test_df['product']):.3f}")
    report(test_df["product"], clf.predict(X_test), PRODUCTS, name="baseline")

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_DIR.parent / "baseline_logreg.joblib")


if __name__ == "__main__":
    main()
