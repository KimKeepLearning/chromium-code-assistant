"""Shared config + helpers for all scripts. Reads config.yaml from repo root."""
import os
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config():
    with open(os.path.join(_ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    # expand ~ in the repo path
    cfg["repo"]["path"] = os.path.expanduser(cfg["repo"]["path"])
    return cfg


def root_path(*parts):
    """Path relative to the repo root (where config.yaml lives)."""
    return os.path.join(_ROOT, *parts)


def data_path(*parts):
    return root_path("data", *parts)
