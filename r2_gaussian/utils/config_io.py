"""Versioned argument snapshots, with a non-executing legacy reader."""
import ast
import json
import locale
import os
from pathlib import Path
import tempfile


def parse_legacy_namespace(text):
    expression = ast.parse(text, mode="eval").body
    if (not isinstance(expression, ast.Call)
            or not isinstance(expression.func, ast.Name)
            or expression.func.id != "Namespace" or expression.args):
        raise ValueError("Legacy configuration must be Namespace(literal_keyword=value, ...)")
    result = {}
    for keyword in expression.keywords:
        if keyword.arg is None or keyword.arg in result:
            raise ValueError("Legacy configuration has expanded or duplicate keywords")
        try:
            result[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Nonliteral legacy argument: {keyword.arg}") from error
    return result


def saved_config_path(model_path):
    root = Path(model_path)
    for name in ("cfg_args.json", "cfg_args"):
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No saved argument configuration in {root}")


def load_saved_config(model_path):
    path = saved_config_path(model_path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        if path.suffix == ".json":
            raise
        # Older Windows loggers used the process locale for cfg_args.
        text = path.read_text(encoding=locale.getpreferredencoding(False))
    if path.suffix != ".json":
        return parse_legacy_namespace(text)
    def reject_constant(value):
        raise ValueError(f"Non-finite saved argument: {value}")
    payload = json.loads(text, parse_constant=reject_constant)
    if (not isinstance(payload, dict) or payload.get("format_version") != 1
            or not isinstance(payload.get("arguments"), dict)):
        raise ValueError(f"Unsupported saved argument configuration: {path}")
    return payload["arguments"]


def write_saved_config(model_path, arguments):
    root = Path(model_path)
    root.mkdir(parents=True, exist_ok=True)
    # Only Paths need conversion; unsupported values should fail explicitly.
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Unsupported saved configuration value: {type(value).__name__}")
    payload = json.dumps({"format_version": 1, "arguments": dict(arguments)},
                         indent=2, ensure_ascii=False, allow_nan=False, default=convert)
    descriptor, temporary = tempfile.mkstemp(dir=root, suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload + "\n")
        os.replace(temporary, root / "cfg_args.json")
    finally:
        Path(temporary).unlink(missing_ok=True)
