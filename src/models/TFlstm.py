import datetime

import numpy as np
import tensorflow as tf
from src.models.abstract import Model


class TFlstm(Model):
    def __init__(self, dataset=None):
        super(TFlstm, self).__init__(
            dataset=dataset,
        )

    def summary(self):
        self.model.summary()

    def make_model(
        self, vocab_size=0, output_dim=512, embeddings_initializer="uniform"
    ):
        trainable_embedding = True
        if embeddings_initializer != "uniform":
            trainable_embedding = False
        input_shape = self.dataset.input_shape
        self.model = tf.keras.models.Sequential(
            [
                # Layer Input Word Embedding
                tf.keras.layers.Embedding(
                    vocab_size + 1,
                    output_dim=output_dim,
                    input_shape=[
                        input_shape,
                    ],
                    embeddings_initializer=embeddings_initializer,
                    trainable=trainable_embedding,
                ),
                tf.keras.layers.SpatialDropout1D(0.2),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)
                ),
                # Layer Dense classique
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.params["lr"]),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

    def fit(self, validation_data=None):
        batched_data = self.dataset.make_tf_batched_data(self.params["batch_size"])
        if validation_data:
            batched_val = validation_data.make_tf_batched_data(
                self.params["batch_size"]
            )
        else:
            batched_val = None

        log_dir = f"models/logs/{self.name}" + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        self.model.fit(
            batched_data,
            epochs=self.params["epochs"],
            validation_data=batched_val,
            callbacks=[tensorboard_callback],
        )

    def predict(self):
        predictions = self.model.predict(self.dataset._features)
        return tf.squeeze(predictions)

    def save(self):
        self.model.save(f"models/{self.name}/model.h5")

    def load(self):
        self.model = tf.keras.models.load_model(f"models/{self.name}/model.h5")
