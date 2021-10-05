import json
from pathlib import Path


# TODO: Implement
def prepare():

    out_path = Path("data/prepared")
    out_path.mkdir(parents=True, exist_ok=True)

    for out in [
        "train.csv",
        "validate.csv",
        "test.csv",
    ]:
        open(out_path / out, "w").write("")


if __name__ == "__main__":
    print("I am preparing the data !")
    prepare()
