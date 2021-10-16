import joblib
from sklearn.linear_model import LogisticRegression
from src.models.abstract import Model


class SKLogisticRegression(Model):
    def __init__(self, dataset=None):
        super(SKLogisticRegression, self).__init__(
            dataset=dataset,
        )

    def make_model(self, vocab_size=0):
        self.model = LogisticRegression(max_iter=100)

    def fit(self, validation_data=None):
        self.model.fit(self.dataset._features, self.dataset._labels)

    def save(self):
        joblib.dump(self.model, f"models/{self.name}/model.joblib")

    def load(self):
        self.model = joblib.load(f"models/{self.name}/model.joblib")
