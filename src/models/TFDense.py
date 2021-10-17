import tensorflow as tf
from src.models.abstract import Model


class TFDense(Model):
    def __init__(self, dataset=None):
        super(TFDense, self).__init__(
            dataset=dataset,
        )

    def summary(self):
        self.model.summary()

    def make_model(self, vocab_size=0):
        input_shape = self.dataset.input_shape
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(64, input_shape=[input_shape], activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation="relu"),
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
        self.model.fit(
            batched_data, epochs=self.params["epochs"], validation_data=batched_val
        )

    def predict(self):
        predictions = super().predict()
        return tf.squeeze(predictions)

    def save(self):
        self.model.save(f"models/{self.name}/model.h5")

    def load(self):
        self.model = tf.keras.models.load_model(f"models/{self.name}/model.h5")
