# config.py
import yaml

def load_config(path="config/config.yaml"):
    """
    Load a YAML configuration file.

    Args:
        path (str): Path to YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg