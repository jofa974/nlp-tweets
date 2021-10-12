import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from logger import logger


class SKLogisticRegression:
    def __init__(self, vocab_size, input_shape):
        self.model = LogisticRegression(max_iter=100)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


class TFConv1D:
    def __init__(self, vocab_size, input_shape):
        self.model = tf.keras.Sequential(
            [
                # Layer Input Word Embedding
                tf.keras.layers.Embedding(
                    vocab_size + 1,
                    output_dim=64,
                    input_shape=[
                        input_shape,
                    ],
                ),
                tf.keras.layers.Conv1D(16, 3, activation="relu"),
                # Flatten
                tf.keras.layers.Flatten(),
                # Layer Dense classique
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                # Output layer with number of output neurons equal to class number with softmax function
                tf.keras.layers.Dense(5, activation="softmax"),
            ]
        )

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


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
factory.register_builder("logistic_regression", SKLogisticRegression)
factory.register_builder("tf_conv1d", TFConv1D)
