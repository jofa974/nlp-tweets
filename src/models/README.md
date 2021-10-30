# Models

A new model can be implemented as follows:

- Define a class inheriting from `Model` or `TFModel` in `<new_model.py>`
- Register the new model in the factory located in `__init__.py`
- Add the model information (preprocessor, output name) in `dvc.yaml`
