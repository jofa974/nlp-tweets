import tensorflow as tf
from src.models.abstract import TFModel


class TFConv1D(TFModel):
    def __init__(self):
        super().__init__()

    def make_model(self, input_shape, vocab_size=0):
        self.model = tf.keras.models.Sequential(
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
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        lr = self.lr_scheduler(self.params["lr"])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )
