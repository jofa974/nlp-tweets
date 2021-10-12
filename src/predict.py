import argparse
from pathlib import Path


def predict(model_name):

    out_path = Path("data/predictions")
    out_path.mkdir(parents=True, exist_ok=True)

    open(out_path / f"{model_name}_prediction.csv", "w").write("")


if __name__ == "__main__":
    print("I am making prediction !")

    parser = argparse.ArgumentParser(description="Prediction")
    parser.add_argument(
        "--model-name",
        type=str,
        default="logistic_regression",
        help="A model name. Must be a class registered in src/models.py:factory",
    )

    args = parser.parse_args()
    predict(args.model_name)
