from typing import Callable, Optional
import numpy as np

def tiled_inference(
    model,
    array: np.ndarray,
    tile_size: int = 64,
    overlap: int = 32,
    scale: int = 4,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """
    Runs tiled inference over multi-band arrays (C, H, W) or (H, W).
    Guarantees every tile passed to the model has dimensions (tile_size, tile_size).
    """
    is_2d = array.ndim == 2
    if is_2d:
        array = array[np.newaxis, ...]

    c, h, w = array.shape
    stride = tile_size - overlap

    out_h = h * scale
    out_w = w * scale
    output = np.zeros((c, out_h, out_w), dtype=np.float32)
    weight_map = np.zeros((out_h, out_w), dtype=np.float32)

    y_starts = list(range(0, max(1, h - overlap), stride))
    x_starts = list(range(0, max(1, w - overlap), stride))

    # Clamp the last start positions to guarantee full tile_size
    y_starts = [min(y, max(0, h - tile_size)) for y in y_starts]
    x_starts = [min(x, max(0, w - tile_size)) for x in x_starts]
    y_starts = sorted(list(set(y_starts)))
    x_starts = sorted(list(set(x_starts)))

    total_tiles = len(y_starts) * len(x_starts)
    processed = 0

    # Linear blend weight window to eliminate seam artifacts
    window_1d = np.bartlett(tile_size * scale) + 1e-3
    blend_window = np.outer(window_1d, window_1d)

    for y in y_starts:
        for x in x_starts:
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)

            tile = array[:, y:y_end, x:x_end]
            th, tw = tile.shape[1], tile.shape[2]

            # If edge raster is smaller than tile_size, pad to full tile_size
            pad_h = tile_size - th
            pad_w = tile_size - tw
            if pad_h > 0 or pad_w > 0:
                tile = np.pad(tile, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

            # Inference
            inp = tile[0] if is_2d else tile
            pred = np.asarray(model.predict(inp))
            if pred.ndim == 2:
                pred = pred[np.newaxis, ...]

            # Crop off any padding added prior to inference
            pred = pred[:, :th * scale, :tw * scale]
            w_win = blend_window[:th * scale, :tw * scale]

            out_y = y * scale
            out_x = x * scale
            out_ye = out_y + th * scale
            out_xe = out_x + tw * scale

            output[:, out_y:out_ye, out_x:out_xe] += pred * w_win
            weight_map[out_y:out_ye, out_x:out_xe] += w_win

            processed += 1
            if progress_callback:
                progress_callback(processed, total_tiles)

    # Normalize blended overlapping areas
    weight_map = np.maximum(weight_map, 1e-6)
    output /= weight_map[np.newaxis, ...]

    return output[0] if is_2d else output