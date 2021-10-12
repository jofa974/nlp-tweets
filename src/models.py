import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from logger import logger


class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=100)


class ModelFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


factory = ModelFactory()
factory.register_builder("logistic_regression", LogisticRegression)
