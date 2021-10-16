from abc import ABC, abstractmethod

import yaml


class Model(ABC):
    def __init__(self, dataset=None):
        self.model = None
        self.dataset = dataset
        self.name = self.__class__.__name__
        self.get_params()

    @abstractmethod
    def make_model(self, vocab_size=0):
        raise NotImplementedError

    @abstractmethod
    def fit(self, validation_data=None):
        raise NotImplementedError

    def predict(self):
        predictions = self.model.predict(self.dataset._features)
        threshold = 0.5
        return list(map((lambda x: 1 if x > threshold else 0), predictions))

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    def get_params(self):
        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)[self.name]
