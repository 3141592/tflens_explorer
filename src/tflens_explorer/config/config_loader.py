from pathlib import Path
import yaml

def load_model_aliases():
    # go up from this file → project root → config/models.yaml
    base_dir = Path(__file__).resolve().parents[3]
    config_path = base_dir / "config" / "models.yaml"

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return data.get("models", {})


def load_model_evals():
    # go up from this file → project root → config/evals.yaml
    base_dir = Path(__file__).resolve().parents[3]
    config_path = base_dir / "config" / "evals.yaml"

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return data


def load_tflens_internals():
    # go up from this file → project root → config/models.yml
    base_dir = Path(__file__).resolve().parents[3]
    config_path = base_dir / "config" / "internals.yaml"

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return data



