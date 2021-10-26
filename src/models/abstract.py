import datetime
from abc import ABC, abstractclassmethod, abstractmethod

import tensorflow as tf
import yaml


class Model(ABC):
    def __init__(self):
        self.model = None
        self.name = self.__class__.__name__
        self.get_params()

    @abstractmethod
    def summary(self):
        raise NotImplementedError

    @abstractmethod
    def make_model(self, vocab_size=0):
        raise NotImplementedError

    @abstractmethod
    def fit(self, dataset, use_validation=False):
        raise NotImplementedError

    @abstractmethod
    def predict(self, dataset):
        raise NotImplementedError

    def predict_class(self, dataset, threshold):
        predictions = self.predict(dataset)
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

    @abstractclassmethod
    def from_preprocessor(cls, preproc, input_shape):
        """Build an instance of this class based on a preprocessor data.

        Args:
            preproc (src.preprocessors.Preprocessor): A document Preprocessor class instance
        """
        raise NotImplementedError


class TFModel(Model):
    def __init__(self):
        super().__init__()

    def summary(self):
        self.model.summary()

    @abstractmethod
    def make_model(self, input_shape, vocab_size=0):
        raise NotImplementedError

    def fit(self, dataset, use_validation=False):
        if use_validation:
            train_ds, val_ds = dataset.train_test_split()
            batched_val = val_ds.make_tf_batched_data(self.params["batch_size"])
            batched_train = train_ds.make_tf_batched_data(self.params["batch_size"])
        else:
            batched_val = None
            batched_train = dataset.make_tf_batched_data(self.params["batch_size"])
        log_dir = f"models/logs/{self.name}" + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        self.model.fit(
            batched_train,
            epochs=self.params["epochs"],
            validation_data=batched_val,
            callbacks=[tensorboard_callback, LearningRateLogger()],
        )

    def predict(self, dataset):
        predictions = self.model.predict(dataset._features)
        return tf.squeeze(predictions)

    def save(self):
        self.model.save(f"models/{self.name}/model.h5")

    def load(self):
        self.model = tf.keras.models.load_model(f"models/{self.name}/model.h5")

    @classmethod
    def from_preprocessor(cls, preproc, input_shape=0):
        """Build an instance of this class based on a preprocessor data.

        Args:
            preproc (src.preprocessors.Preprocessor): A document Preprocessor class instance
        """
        instance = cls()
        instance.make_model(input_shape, preproc.vocab_size)
        return instance

    def lr_scheduler(self, initial_learning_rate):
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=100, decay_rate=0.96, staircase=True
        )


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        logs["learning_rate"] = self.model.optimizer.lr
