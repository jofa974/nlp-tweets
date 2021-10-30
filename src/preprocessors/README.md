# Preprocessors

A new data preprocessor can be implemented as follows:

- Define a class inheriting from `Preprocessor` in `<new_model.py>`
- Register the new preprocessor in the factory located in `__init__.py`
- Add the preprocessor information in `dvc.yaml` (in particular the `prepare_data` stage)