import csv
import math
import random
from typing import List, Tuple, Dict
import os
from matplotlib import pyplot as plt

# ============================================
# Modul ini melakukan:
# - Preprocessing ringan: load CSV, split train/test
# - Pengelompokkan (classification): Gaussian Naive Bayes
# - Evaluasi: akurasi & confusion matrix
# Catatan: Tidak ada "clustering" (pengelompokan tanpa label) di skrip ini.
# ============================================


class GaussianNB:
    # --------------------------------------------
    # Komponen Pengelompokkan: Model Gaussian Naive Bayes
    # - Menyimpan prior kelas, mean (theta), dan varians (sigma)
    # - Asumsi fitur independen bersyarat dan berdistribusi Gaussian
    # --------------------------------------------
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes: List[int] = []
        self.class_prior: Dict[int, float] = {}
        self.theta: Dict[int, List[float]] = {}
        self.sigma: Dict[int, List[float]] = {}

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        # --------------------------------------------
        # Tahap Training (Pengelompokkan)
        # - Hitung prior kelas
        # - Hitung mean dan varians per fitur per kelas
        # --------------------------------------------
        n = len(X)
        d = len(X[0]) if n > 0 else 0
        counts: Dict[int, int] = {}
        sums: Dict[int, List[float]] = {}
        sq_sums: Dict[int, List[float]] = {}
        for xi, yi in zip(X, y):
            if yi not in counts:
                counts[yi] = 0
                sums[yi] = [0.0] * d
                sq_sums[yi] = [0.0] * d
            counts[yi] += 1
            for j in range(d):
                v = xi[j]
                sums[yi][j] += v
                sq_sums[yi][j] += v * v
        self.classes = sorted(counts.keys())
        total = float(n)
        for c in self.classes:
            self.class_prior[c] = counts[c] / total
            mu = [s / counts[c] for s in sums[c]]
            var = []
            for j in range(d):
                m = mu[j]
                v = (sq_sums[c][j] / counts[c]) - (m * m)
                if v <= 0:
                    v = self.var_smoothing
                else:
                    v += self.var_smoothing
                var.append(v)
            self.theta[c] = mu
            self.sigma[c] = var

    def _log_gaussian(self, x: float, mu: float, var: float) -> float:
        return -0.5 * math.log(2.0 * math.pi * var) - ((x - mu) ** 2) / (2.0 * var)

    def predict_one(self, x: List[float]) -> int:
        # --------------------------------------------
        # Inferensi (Pengelompokkan)
        # - Hitung log-likelihood per kelas
        # - Pilih kelas dengan probabilitas posterior tertinggi
        # --------------------------------------------
        best_c = None
        best_logp = -float("inf")
        for c in self.classes:
            logp = math.log(self.class_prior[c])
            mu = self.theta[c]
            var = self.sigma[c]
            for j in range(len(x)):
                logp += self._log_gaussian(x[j], mu[j], var[j])
            if logp > best_logp:
                best_logp = logp
                best_c = c
        if best_c is None:
            raise ValueError("Model belum dilatih atau tidak ada kelas yang tersedia.")
        return int(best_c)

    def predict(self, X: List[List[float]]) -> List[int]:
        return [self.predict_one(x) for x in X]


def load_csv(path: str) -> Tuple[List[List[float]], List[int], List[str]]:
    # --------------------------------------------
    # Preprocessing: Memuat data dari CSV
    # - Memisahkan fitur (13 kolom pertama) dan target (kolom terakhir)
    # --------------------------------------------
    X: List[List[float]] = []
    y: List[int] = []
    headers: List[str] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            vals = [float(v) for v in row]
            X.append(vals[:-1])
            y.append(int(vals[-1]))
    return X, y, headers


def train_test_split(X: List[List[float]], y: List[int], test_size: float = 0.2, seed: int = 42) -> Tuple[List[List[float]], List[List[float]], List[int], List[int]]:
    # --------------------------------------------
    # Preprocessing: Membagi data menjadi train/test
    # - Split 80/20 dengan pengacakan deterministik (seed)
    # --------------------------------------------
    n = len(X)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = int(n * (1.0 - test_size))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true) if y_true else 0.0


def confusion_matrix(y_true: List[int], y_pred: List[int]) -> Dict[Tuple[int, int], int]:
    m: Dict[Tuple[int, int], int] = {}
    for t, p in zip(y_true, y_pred):
        key = (t, p)
        m[key] = m.get(key, 0) + 1
    return m


def save_confusion_matrix(cm: Dict[Tuple[int, int], int]) -> None:
    w = [
        [cm.get((0, 0), 0), cm.get((0, 1), 0)],
        [cm.get((1, 0), 0), cm.get((1, 1), 0)],
    ]
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(w, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(w[i][j]), ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    os.makedirs("outputs", exist_ok=True)
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_metrics(features: int, train_size: int, test_size: int, accuracy_value: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6, 4))
    def draw(ax, label: str, value: str) -> None:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{label}: {value}", ha="center", va="center", fontsize=16)
    draw(axes[0][0], "Features", str(features))
    draw(axes[0][1], "Accuracy", f"{accuracy_value:.4f}")
    draw(axes[1][0], "Train size", str(train_size))
    draw(axes[1][1], "Test size", str(test_size))
    os.makedirs("outputs", exist_ok=True)
    fig.suptitle("Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig("outputs/metrics.png", dpi=160)
    plt.close(fig)


def main() -> None:
    # --------------------------------------------
    # Pipeline Eksekusi:
    # 1) Preprocessing: load CSV & train/test split
    # 2) Pengelompokkan: inisialisasi & training GaussianNB
    # 3) Evaluasi: akurasi & confusion matrix
    # --------------------------------------------
    X, y, headers = load_csv("heart.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    model = GaussianNB(var_smoothing=1e-9)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("Features:", len(X[0]))
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    print("Accuracy:", round(acc, 4))
    print("Confusion matrix entries:")
    for key in sorted(cm.keys()):
        print(str(key) + ":" + str(cm[key]))
    save_confusion_matrix(cm)
    save_metrics(len(X[0]), len(X_train), len(X_test), acc)


if __name__ == "__main__":
    main()
