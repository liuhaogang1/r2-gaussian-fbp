import yaml
import json
import os.path as osp
from pathlib import Path


def args2string(hp):
    """
    Transfer args to string.
    """
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))


def load_config(path, default_path=None, _seen=()):
    """Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    """
    # Load configuration from file itself
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Cyclic config inheritance: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg_special = yaml.safe_load(f) or {}
    if not isinstance(cfg_special, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")

    # Check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        parent = Path(inherit_from)
        if not parent.is_absolute():
            parent = path.parent / parent
        cfg = load_config(parent, default_path, (*_seen, path))
    elif default_path is not None:
        cfg = load_config(default_path, _seen=(*_seen, path))
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
