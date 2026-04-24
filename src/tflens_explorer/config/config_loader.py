from pathlib import Path
import yaml

def load_model_aliases():
    # go up from this file → project root → config/models.yml
    base_dir = Path(__file__).resolve().parents[3]
    config_path = base_dir / "config" / "models.yml"

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return data.get("models", {})