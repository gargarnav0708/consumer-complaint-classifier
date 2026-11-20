"""Quick exploratory plots of the complaints data, saved to reports/figures/.

Run:  python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR
from src.data import filter_complaints, load_complaints


def save_barh(counts, title, filename):
    """Horizontal bar chart of a value_counts() Series."""
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.sort_values().plot.barh(ax=ax, color="#F1A424")
    ax.set_title(title)
    ax.set_xlabel("Number of complaints")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = filter_complaints(load_complaints(extra_columns=["Issue", "Company"]))
    print(f"{len(df)} complaints between {df['date'].min():%Y-%m-%d} and {df['date'].max():%Y-%m-%d}")

    save_barh(df["product"].value_counts(), "Complaints per product", "products.png")
    save_barh(df["issue"].value_counts().head(15), "Top 15 complaint issues", "issues.png")
    save_barh(df["company"].value_counts().head(10), "Top 10 companies by complaints", "companies.png")

    # Weekly trend
    weekly = df.set_index("date").resample("W").size()
    fig, ax = plt.subplots(figsize=(12, 4))
    weekly.plot(ax=ax, color="#F1A424")
    ax.set_title("Complaints per week")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "weekly_trend.png")
    plt.close(fig)

    print(f"Plots saved -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
