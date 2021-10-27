import numpy as np
import tensorflow as tf
from src.models.abstract import TFModel


class TFlstm(TFModel):
    def __init__(self):
        super().__init__()

    def make_model(self, input_shape, embedding_layer):
        """Constructs and compiles a DNN model.

        Args:
            input_shape (int): Input shape
            embedding_layer (tf.keras.layers.Embedding): an embedding layer produced by a Preprocessor object.
        """
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(
                    input_shape=[
                        input_shape,
                    ]
                ),
                # Layer Input Word Embedding
                embedding_layer,
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(128, dropout=0.4, return_sequences=True)
                ),
                tf.keras.layers.Conv1D(64, 5, activation="relu"),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(128, dropout=0.4, return_sequences=True)
                ),
                tf.keras.layers.LSTM(64, dropout=0.4),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        lr = self.lr_scheduler(self.params["lr"])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

    @classmethod
    def from_preprocessor(cls, preproc, input_shape):
        """Build an instance of this class based on a preprocessor data.

        Args:
            preproc (src.preprocessors.Preprocessor): A document Preprocessor class instance
            input_shape (int): the input shape
        """
        instance = cls()
        instance.make_model(
            input_shape=input_shape, embedding_layer=preproc.make_tf_embedding_layer()
        )
        return instance
