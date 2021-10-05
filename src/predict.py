from pathlib import Path


# TODO: Implement
def predict():

    out_path = Path("data/predictions")
    out_path.mkdir(parents=True, exist_ok=True)

    open(out_path / "simple_model_prediction.csv", "w").write("")


if __name__ == "__main__":
    print("I am making prediction !")
    predict()
