from pathlib import Path
import json
from typing import Any, Dict

import numpy as np


def load_saved_artifacts(output_dir: str) -> Dict[str, Any]:
    """Read saved NumPy/JSON artifacts without running pipeline logic."""
    directory = Path(output_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {directory}")
    artifacts: Dict[str, Any] = {}
    for path in sorted(directory.rglob("*.npy")):
        artifacts[str(path.relative_to(directory))] = np.load(path, allow_pickle=False)
    for path in sorted(directory.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as values:
            artifacts[str(path.relative_to(directory))] = {
                key: values[key] for key in values.files
            }
    for path in sorted(directory.rglob("*.json")):
        artifacts[str(path.relative_to(directory))] = json.loads(
            path.read_text(encoding="utf-8")
        )
    return artifacts


def summarize_artifacts(output_dir: str) -> Dict[str, Any]:
    """Return artifact names and shapes for a lightweight dashboard summary."""
    summary = {}
    for name, artifact in load_saved_artifacts(output_dir).items():
        if isinstance(artifact, np.ndarray):
            summary[name] = tuple(artifact.shape)
        elif isinstance(artifact, dict):
            summary[name] = {
                key: tuple(value.shape) if isinstance(value, np.ndarray) else value
                for key, value in artifact.items()
            }
        else:
            summary[name] = type(artifact).__name__
    return summary
