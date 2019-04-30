import os
from joblib import dump

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, f1_score


MODEL_PATH = os.path.join('jar', 'iris_classifier.joblib')


def main():
    iris = load_iris()
    labels = iris.target_names
    X = iris['data']
    y = iris['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = RandomForestClassifier(
        n_estimators=10,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)

    results = confusion_matrix(y_test, y_hat)

    print("Confusion Matrix:")
    df = pd.DataFrame(results, columns=labels)
    df.set_index(labels, inplace=True)
    print(df.to_string(justify='center'), "\n")

    accuracy = accuracy_score(y_test, y_hat)
    recall = recall_score(y_test, y_hat, average='macro')
    f1 = f1_score(y_test, y_hat, average='macro')

    print(f"Accuracy: {100 * accuracy:.2f}%")
    print(f"Recall: {100 * recall:.2f}%")
    print(f"F1-Score: {100 * f1:.2f}%")

    dump(model, MODEL_PATH)


if __name__ == "__main__":
    main()
