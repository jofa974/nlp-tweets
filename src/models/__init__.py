import os
import random

import numpy as np
import tensorflow as tf
from src.models.SKLogisticRegression import SKLogisticRegression
from src.models.TFConv1D import TFConv1D

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # new flag present in tf 2.0+
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class ModelFactory:
    def __init__(self):
        self._creators = {}

    def register_model(self, model, creator):
        self._creators[model] = creator

    def get_model(self, format, *args, **kwargs):
        creator = self._creators.get(format)
        if not creator:
            raise ValueError(format)
        return creator(*args, **kwargs)


model_factory = ModelFactory()
model_factory.register_model("SKLogisticRegression", SKLogisticRegression)
model_factory.register_model("TFConv1D", TFConv1D)
