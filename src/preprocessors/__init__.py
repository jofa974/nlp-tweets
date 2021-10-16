from .SKCountVectorizer import SKCountVectorizer
from .TFTokenizer import TFTokenizer

# TODO: use factory
constructors = {"SKCountVectorizer": SKCountVectorizer, "TFTokenizer": TFTokenizer}
