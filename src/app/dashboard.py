from pathlib import Path
from typing import Dict

import numpy as np


def load_saved_artifacts(output_dir: str) -> Dict[str, np.ndarray]:
    """Read saved NumPy artifacts from data/outputs without running pipeline logic."""
    directory = Path(output_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {directory}")
    artifacts: Dict[str, np.ndarray] = {}
    for path in sorted(directory.rglob("*.npy")):
        artifacts[str(path.relative_to(directory))] = np.load(path, allow_pickle=False)
    return artifacts


def summarize_artifacts(output_dir: str) -> Dict[str, tuple]:
    """Return artifact names and shapes for a lightweight dashboard summary."""
    return {name: tuple(array.shape) for name, array in load_saved_artifacts(output_dir).items()}
