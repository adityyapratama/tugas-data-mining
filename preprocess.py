import csv
import os
import random
from typing import List, Tuple
from matplotlib import pyplot as plt


def load_rows(path: str) -> Tuple[List[str], List[List[str]]]:
    headers: List[str] = []
    rows: List[List[str]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            rows.append(row)
    return headers, rows


def split_rows(rows: List[List[str]], test_size: float = 0.2, seed: int = 42) -> Tuple[List[List[str]], List[List[str]]]:
    n = len(rows)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = int(n * (1.0 - test_size))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    return train_rows, test_rows


def save_csv(path: str, headers: List[str], rows: List[List[str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> None:
    headers, rows = load_rows("heart.csv")
    total = len(rows)
    features = len(headers) - 1
    train_rows, test_rows = split_rows(rows, test_size=0.2, seed=42)
    print("Total data:", total)
    print("Features:", features)
    print("Train size:", len(train_rows))
    print("Test size:", len(test_rows))
    print("Headers:", ",".join(headers))
    save_csv("outputs/train.csv", headers, train_rows)
    save_csv("outputs/test.csv", headers, test_rows)
    def counts(rows: List[List[str]]) -> dict:
        c = {}
        for r in rows:
            lbl = int(r[-1])
            c[lbl] = c.get(lbl, 0) + 1
        return c
    train_counts = counts(train_rows)
    test_counts = counts(test_rows)
    labels = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
    x = list(range(len(labels)))
    train_vals = [train_counts.get(l, 0) for l in labels]
    test_vals = [test_counts.get(l, 0) for l in labels]
    width = 0.4
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([i - width / 2 for i in x], train_vals, width=width, label="Train")
    ax.bar([i + width / 2 for i in x], test_vals, width=width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    ax.set_title("Train/Test Split Distribution by Target")
    ax.legend()
    os.makedirs("outputs", exist_ok=True)
    fig.tight_layout()
    fig.savefig("outputs/train_test_split.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
