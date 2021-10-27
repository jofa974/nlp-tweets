from src.preprocessors.GloVeVectorizer import GloVeVectorizer
from src.preprocessors.SKCountVectorizer import SKCountVectorizer
from src.preprocessors.SKTfidfVectorizer import SKTfidfVectorizer
from src.preprocessors.TFTokenizer import TFTokenizer


class PreprocessorFactory:
    def __init__(self):
        self._creators = {}

    def register_method(self, method, creator):
        self._creators[method] = creator

    def get_preprocessor(self, format):
        creator = self._creators.get(format)
        if not creator:
            raise ValueError(format)
        return creator()


preprocessor_factory = PreprocessorFactory()
preprocessor_factory.register_method("SKCountVectorizer", SKCountVectorizer)
preprocessor_factory.register_method("SKTfidfVectorizer", SKTfidfVectorizer)
preprocessor_factory.register_method("TFTokenizer", TFTokenizer)
preprocessor_factory.register_method("GloVeVectorizer", GloVeVectorizer)
