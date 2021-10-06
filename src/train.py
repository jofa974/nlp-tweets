import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def train():

    train = np.loadtxt("data/prepared/train.csv", delimiter=",")
    X_train = train[:, :-1]
    Y_train = train[:, -1]

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    Y_train_pred = model.predict(X_train)
    metrics = {"f1_score": f1_score(y_true=Y_train, y_pred=Y_train_pred)}

    with open("models/train_metrics.json", "w") as f:
        json.dump(metrics, f)

    joblib.dump(model, "models/simple_model.joblib")


if __name__ == "__main__":
    from logger import setup_applevel_logger

    log = setup_applevel_logger(file_name="app_debug.log")

    log.info("I am training !")
    train()
