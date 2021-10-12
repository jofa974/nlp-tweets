import joblib
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

from logger import logger


class SKLogisticRegression:
    def __init__(self, **kwargs):
        self.model = None

    def make_model(self):
        self.model = LogisticRegression(max_iter=100)

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, model_name):
        joblib.dump(self.model, f"models/{model_name}/model.joblib")

    def load(self, model_name):
        self.model = joblib.load(f"models/{model_name}/model.joblib")


class TFConv1D:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 0)
        self.input_shape = kwargs.get("input_shape", 1)
        self.lr = kwargs.get("lr", 1e-1)

    def make_model(self):
        self.model = tf.keras.Sequential(
            [
                # Layer Input Word Embedding
                tf.keras.layers.Embedding(
                    self.vocab_size + 1,
                    output_dim=512,
                    input_shape=[
                        self.input_shape,
                    ],
                ),
                tf.keras.layers.Conv1D(128, 3, activation="relu"),
                # Flatten
                tf.keras.layers.Flatten(),
                # Layer Dense classique
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                # Output layer with number of output neurons equal to class number with softmax function
                tf.keras.layers.Dense(1, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

    def fit(self, X, Y):
        data = tf.data.Dataset.from_tensor_slices((X, Y))
        data = data.batch(64)
        self.model.fit(data, epochs=50)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, model_name):
        self.model.save(f"models/{model_name}/model.h5")

    def load(self, model_name):
        self.model = tf.keras.models.load_model(f"models/{model_name}/model.h5")


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
