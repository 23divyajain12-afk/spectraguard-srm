from typing import Callable, Optional

import numpy as np

from src.core.schemas import SRModel


def tiled_inference(
    model: SRModel,
    image: np.ndarray,
    tile_size: int,
    overlap: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Run SR inference over overlapping tiles and average their overlaps."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim not in (2, 3):
        raise ValueError("image must have shape (H,W) or (C,H,W)")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size must be positive and overlap must be smaller")

    channels = array[None, ...] if array.ndim == 2 else array
    height, width = channels.shape[-2:]
    step = tile_size - overlap
    total_tiles = len(range(0, height, step)) * len(range(0, width, step))
    completed_tiles = 0
    output = None
    weights = None
    for row in range(0, height, step):
        for col in range(0, width, step):
            row_stop = min(row + tile_size, height)
            col_stop = min(col + tile_size, width)
            tile = channels[:, row:row_stop, col:col_stop]
            prediction = np.asarray(model.predict(tile[0] if array.ndim == 2 else tile))
            predicted = prediction[None, ...] if prediction.ndim == 2 else prediction
            if predicted.shape[0] != channels.shape[0]:
                raise ValueError("model output channel count does not match input")
            scale_y = predicted.shape[1] / tile.shape[1]
            scale_x = predicted.shape[2] / tile.shape[2]
            if abs(scale_y - scale_x) > 1e-6:
                raise ValueError("model output must use the same scale on both axes")
            if output is None:
                output = np.zeros(
                    (channels.shape[0], round(height * scale_y), round(width * scale_x)),
                    dtype=np.float32,
                )
                weights = np.zeros(output.shape[1:], dtype=np.float32)
            out_row = round(row * scale_y)
            out_col = round(col * scale_x)
            output[:, out_row:out_row + predicted.shape[1], out_col:out_col + predicted.shape[2]] += predicted
            weights[out_row:out_row + predicted.shape[1], out_col:out_col + predicted.shape[2]] += 1
            completed_tiles += 1
            if progress_callback is not None:
                progress_callback(completed_tiles, total_tiles)

    if output is None or weights is None or np.any(weights == 0):
        raise ValueError("tiled inference produced incomplete output")
    result = output / weights[None, ...]
    return result[0] if array.ndim == 2 else result
