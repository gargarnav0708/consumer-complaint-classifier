"""Load, clean and split the consumer complaints data.

Pipeline:  raw CSV -> keep chosen products & years -> cap each class -> train/test split
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, MAX_SAMPLES_PER_CLASS, PRODUCTS, SEED, TEST_SIZE, YEARS


def load_complaints(path=DATA_PATH, extra_columns=()):
    """Read the CSV, keeping only the columns we need.

    Always returns `date`, `product` and `text`; `extra_columns` (e.g. "Company")
    are added with lower-case names, which is handy for EDA.
    """
    columns = ["Date received", "Product", "Consumer complaint narrative", *extra_columns]
    df = pd.read_csv(path, usecols=columns, low_memory=False)

    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date received": "date", "consumer complaint narrative": "text"})

    df["date"] = pd.to_datetime(df["date"])
    df["product"] = df["product"].str.lower()
    return df.dropna(subset=["text"])


def filter_complaints(df):
    """Keep only the product categories and years we model."""
    keep = df["product"].isin(PRODUCTS) & df["date"].dt.year.isin(YEARS)
    return df[keep].reset_index(drop=True)


def balance_classes(df, max_per_class=MAX_SAMPLES_PER_CLASS):
    """Randomly keep at most `max_per_class` complaints per product."""
    shuffled = df.sample(frac=1, random_state=SEED)
    return shuffled.groupby("product").head(max_per_class).reset_index(drop=True)


def split_data(df):
    """Stratified train/test split, so every product appears in both sets."""
    return train_test_split(
        df[["text", "product"]],
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["product"],
    )


def prepare_data(path=DATA_PATH):
    """Run the full data pipeline and return (train_df, test_df)."""
    df = load_complaints(path)
    df = filter_complaints(df)
    df = balance_classes(df)

    print("Complaints per product:\n", df["product"].value_counts(), "\n")
    train_df, test_df = split_data(df)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows\n")
    return train_df, test_df
