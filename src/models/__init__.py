from src.models.SKLogisticRegression import SKLogisticRegression
from src.models.TFConv1D import TFConv1D


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
