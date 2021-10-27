import joblib
from sklearn.linear_model import LogisticRegression
from src.models.abstract import Model


class SKLogisticRegression(Model):
    def __init__(self):
        super().__init__()

    def summary(self):
        print(self.model.__class__)

    def make_model(self, vocab_size=0):
        self.model = LogisticRegression(max_iter=100)

    def fit(self, dataset, use_validation=False):
        self.model.fit(dataset._features, dataset._labels)

    def predict(self, dataset):
        predictions_proba = self.model.predict_proba(dataset._features)
        # Select only probabilities of belonging to class 1 so that
        # the implementation of predict_class follows the same logic
        # as other model classes.
        return predictions_proba[:, 1]

    def predict_class(self, dataset, threshold):
        predictions = self.predict(dataset)
        return list(map((lambda x: 1 if x > threshold else 0), predictions))

    def save(self):
        joblib.dump(self.model, f"models/{self.name}/model.joblib")

    def load(self):
        self.model = joblib.load(f"models/{self.name}/model.joblib")

    @classmethod
    def from_preprocessor(cls, preproc, input_shape):
        """Build an instance of this class based on a preprocessor data.

        Args:
            preproc (src.preprocessors.Preprocessor): A document Preprocessor class instance
        """
        instance = cls()
        instance.make_model()
        return instance
