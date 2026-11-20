"""Fine-tune DistilBERT to classify complaints into financial products.

The trained model + tokenizer are saved to `models/distilbert-complaints/`,
which is exactly what the Docker image / API loads.

Run:  python -m src.train        (a GPU is strongly recommended)
"""
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import (
    BASE_MODEL, BATCH_SIZE, EPOCHS, ID2LABEL, LABEL2ID, LEARNING_RATE,
    MAX_LENGTH, MODEL_DIR, PRODUCTS, ROOT, WEIGHT_DECAY,
)
from src.data import prepare_data
from src.evaluate import compute_metrics, report


def to_dataset(df, tokenizer):
    """pandas DataFrame -> tokenised Hugging Face Dataset with integer labels."""
    ds = Dataset.from_dict({
        "text": df["text"].tolist(),
        "label": [LABEL2ID[p] for p in df["product"]],
    })
    return ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True,
        remove_columns=["text"],
    )


def main():
    train_df, test_df = prepare_data()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds = to_dataset(train_df, tokenizer)
    test_ds = to_dataset(test_df, tokenizer)

    # Pretrained DistilBERT + a fresh classification head with one output per product
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(PRODUCTS), id2label=ID2LABEL, label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(ROOT / "checkpoints"),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",  # report accuracy / F1 after every epoch
        save_strategy="no",     # we save the final model ourselves below
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tokenizer),  # pads each batch on the fly
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Final evaluation on the held-out test set
    output = trainer.predict(test_ds)
    print(output.metrics)
    y_pred = [ID2LABEL[i] for i in output.predictions.argmax(-1)]
    report(test_df["product"].tolist(), y_pred, PRODUCTS, name="distilbert")

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"Model saved -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
