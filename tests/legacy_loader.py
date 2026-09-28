"""Load selected functions from the original script without executing its model-loading code."""

import ast
from pathlib import Path

import cv2
import numpy as np

LEGACY_SCRIPT = Path(__file__).resolve().parents[1] / "legacy" / "ttc-video_feed.py"


def load_legacy(*names: str, constants: tuple[str, ...] = ()) -> dict:
    tree = ast.parse(LEGACY_SCRIPT.read_text())
    wanted = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in constants for t in node.targets))
    ]
    namespace = {"np": np, "cv2": cv2}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(LEGACY_SCRIPT), "exec"), namespace)
    missing = [n for n in (*names, *constants) if n not in namespace]
    assert not missing, f"not found in legacy script: {missing}"
    return namespace
