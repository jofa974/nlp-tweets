import tensorflow as tf
from src.models.abstract import TFModel


class TFDense(TFModel):
    def __init__(self):
        super().__init__()

    def make_model(self, input_shape, vocab_size=0):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(64, input_shape=[input_shape], activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        lr = self.lr_scheduler(self.params["lr"])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )
