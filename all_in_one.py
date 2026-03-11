from naive_bayes import (
    load_csv,
    train_test_split,
    GaussianNB,
    accuracy,
    confusion_matrix,
    save_confusion_matrix,
    save_metrics,
)


def main() -> None:
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
