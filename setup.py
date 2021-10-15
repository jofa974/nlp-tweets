from setuptools import find_packages, setup

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name="NLPTweets",
    version="0.1.0",
    url="git@github.com:jofa974/nlp-tweets.git",
    author="Jonathan Faustin",
    author_email="faustin.jonathan@gmail.com",
    description="NLP tweets disaster classification",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
)
