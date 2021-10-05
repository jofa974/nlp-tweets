import json


# TODO: Implement
def test():

    with open("models/test_metrics.json", "w") as f:
        json.dump({}, f)


if __name__ == "__main__":
    print("I am testing !")
    test()
