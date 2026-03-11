import os
import json
import pickle
from typing import Dict, Tuple
from naive_bayes import (
    load_csv,
    train_test_split,
    GaussianNB,
    accuracy,
    confusion_matrix,
)


def confusion_matrix_grid(cm: Dict[Tuple[int, int], int]):
    return [
        [cm.get((0, 0), 0), cm.get((0, 1), 0)],
        [cm.get((1, 0), 0), cm.get((1, 1), 0)],
    ]


def main() -> None:
    use_split_files = os.path.exists("outputs/train.csv") and os.path.exists("outputs/test.csv")
    if use_split_files:
        X_train, y_train, headers_train = load_csv("outputs/train.csv")
        X_test, y_test, headers_test = load_csv("outputs/test.csv")
        headers = headers_train
    else:
        X, y, headers = load_csv("heart.csv")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    model = GaussianNB(var_smoothing=1e-9)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Features:", len(X_train[0]))
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    print("Accuracy:", round(acc, 4))
    print("Confusion matrix:", confusion_matrix_grid(cm))

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/gaussian_nb.pkl", "wb") as f:
        pickle.dump(model, f)

    meta = {
        "headers": headers,
        "features": len(headers) - 1,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": acc,
        "confusion_matrix": confusion_matrix_grid(cm),
        "classes": model.classes,
        "class_prior": {str(k): float(v) for k, v in model.class_prior.items()},
        "theta": {str(k): [float(x) for x in v] for k, v in model.theta.items()},
        "sigma": {str(k): [float(x) for x in v] for k, v in model.sigma.items()},
        "var_smoothing": model.var_smoothing,
    }
    with open("outputs/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
