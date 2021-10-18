import joblib
from sklearn.linear_model import LogisticRegression
from src.models.abstract import Model


class SKLogisticRegression(Model):
    def __init__(self, dataset=None):
        super(SKLogisticRegression, self).__init__(
            dataset=dataset,
        )

    def summary(self):
        print(self.model.__class__)

    def make_model(self, vocab_size=0):
        self.model = LogisticRegression(max_iter=100)

    def fit(self, validation_data=None):
        self.model.fit(self.dataset._features, self.dataset._labels)

    def predict(self):
        predictions_proba = self.model.predict_proba(self.dataset._features)
        # Select only probabilities of belonging to class 1 so that
        # the implementation of predict_class follows the same logic
        # as other model classes.
        return predictions_proba[:, 1]

    def predict_class(self, threshold):
        predictions = self.predict()
        return list(map((lambda x: 1 if x > threshold else 0), predictions))

    def save(self):
        joblib.dump(self.model, f"models/{self.name}/model.joblib")

    def load(self):
        self.model = joblib.load(f"models/{self.name}/model.joblib")
