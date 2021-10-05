import json


# TODO: Implement
def train():

    with open("models/train_metrics.json", "w") as f:
        json.dump({}, f)

    with open("models/simple_model.pickle", "w") as f:
        f.write("")


if __name__ == "__main__":
    print("I am training !")
    train()
