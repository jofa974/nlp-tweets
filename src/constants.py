import yaml
import pathlib

module_path = pathlib.Path(__file__).parent.resolve()
with open(module_path / "../params.yaml", "r") as f:
    PARAMS = yaml.safe_load(f)
