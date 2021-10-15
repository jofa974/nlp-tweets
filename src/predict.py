import argparse
from pathlib import Path


def predict(model_class):

    out_path = Path("data/predictions")
    out_path.mkdir(parents=True, exist_ok=True)

    open(out_path / f"{model_class}_prediction.csv", "w").write("")


if __name__ == "__main__":
    print("I am making prediction !")

    parser = argparse.ArgumentParser(description="Prediction")
    parser.add_argument(
        "--model-class",
        type=str,
        default="SKLogisticRegression",
        help="A model class. Must be implemented in a model.py file.",
    )

    args = parser.parse_args()
    predict(args.model_class)
