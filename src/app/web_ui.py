"""Read-only scientific dashboard for existing SpectraGuard-SRM artifacts."""
from __future__ import annotations

import argparse
import html
import io
import json
import math
import re
import struct
import sys
import zlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "config" / "default.yaml"
OUTPUTS = ROOT / "data" / "outputs"
ENDMEMBER_MANIFEST = ROOT / "data" / "raw" / "spectral_library" / "endmember_manifest.json"
SENSOR_RESPONSE = ROOT / "data" / "raw" / "sensor_response" / "COPE-GSEG-EOPG-TN-15-0007 - Sentinel-2 Spectral Response Functions 2024 - 4.0.xlsx"

# SVG Icons Definition
ICONS = {
    "home": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
    "scene": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/></svg>',
    "process": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 7h10"/><path d="M7 12h10"/><path d="M7 17h10"/></svg>',
    "results": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>',
    "analysis": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M15 18a3 3 0 1 0-6 0 3 3 0 0 0 6 0z"/><path d="M13 6v4a2 2 0 0 1-2 2H7"/><path d="M17.5 15.5 22 20"/><path d="M4 14V4a2 2 0 0 1 2-2h10l4 4v7"/></svg>',
    "validation": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.8 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/></svg>',
    "fcls": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/></svg>',
    "about": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
    "sun": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>',
    "expand": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21 21-6-6m6 6v-4.8m0 4.8h-4.8"/><path d="M3 16.2V21m0 0h4.8M3 21l6-6"/><path d="M21 7.8V3m0 0h-4.8M21 3l-6 6"/><path d="M3 7.8V3m0 0h4.8M3 3l6 6"/></svg>',
    "layer": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "globe": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
    "chart": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "shield": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
}

NAV = (
    ("/", "Home", ICONS["home"], "view-home"),
    ("/scene", "Explore Scene", ICONS["scene"], "view-scene"),
    ("/process", "Process", ICONS["process"], "view-process"),
    ("/results", "Results", ICONS["results"], "view-results"),
    ("/analysis", "Analysis", ICONS["analysis"], "view-analysis"),
    ("/validation", "Validation", ICONS["validation"], "view-validation"),
    ("/fcls", "FCLS / Endmembers", ICONS["fcls"], "view-fcls"),
    ("/about", "About", ICONS["about"], "view-about"),
)

ROLE_CANDIDATES = {
    "input": (OUTPUTS / "validation" / "original.npz", OUTPUTS / "validation" / "original.npy"),
    "bicubic": (OUTPUTS / "bicubic" / "bicubic.npz", OUTPUTS / "bicubic" / "bicubic.npy"),
    "sen2sr": (OUTPUTS / "sr" / "spatial_prediction.npz",),
    "spectraguard": (OUTPUTS / "fused" / "fused.npz", OUTPUTS / "fused" / "fused.npy"),
    "metrics": (OUTPUTS / "validation" / "metrics.json", OUTPUTS / "validation" / "metrics.npz"),
    "sam_map": (
        OUTPUTS / "validation" / "sam_map.npy",
        OUTPUTS / "validation" / "metrics.npz",
        OUTPUTS / "uncertainty" / "uncertainty.npz",
    ),
    "uncertainty": (OUTPUTS / "uncertainty" / "uncertainty.npz", OUTPUTS / "uncertainty" / "uncertainty.npy"),
    "abundance": (OUTPUTS / "abundance" / "abundance_sample.npz", OUTPUTS / "abundance" / "abundance.npy"),
    "metadata": (OUTPUTS / "reports" / "run_metadata.json",),
}

GENERATED_PLOTS = {
    "comparison": "Full Spectrum Comparison Overview",
    "comparison_stretched": "Contrast-Stretched Spectral Comparison",
    "ground_detail_comparison": "Ground Detail Structure Comparison",
    "ground_detail_comparison_nir": "NIR Band High-Frequency Detail (B08)",
    "ground_detail_true_color": "True Color Ground Detail (B04/B03/B02)",
}

WAVELENGTHS = {
    "B02": 490,
    "B03": 560,
    "B04": 665,
    "B08": 842,
    "B11": 1610,
}

_IMAGE_CACHE: dict[tuple[str, float], bytes] = {}


def config() -> tuple[dict, str | None]:
    try:
        import yaml

        values = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        return (values if isinstance(values, dict) else {}), None
    except (OSError, ValueError, TypeError) as error:
        return {}, f"Configuration could not be loaded: {error}"


def get_config(data: dict, *keys: str, default: str = "Not available") -> str:
    value: object = data
    try:
        for key in keys:
            value = value[key]  # type: ignore[index]
        return ", ".join(map(str, value)) if isinstance(value, list) else str(value)
    except (KeyError, TypeError):
        return default


def artifact(role: str) -> Path | None:
    return next((path for path in ROLE_CANDIDATES.get(role, ()) if path.is_file()), None)


def inspect_artifact(role: str) -> dict:
    path = artifact(role)
    result = {"path": str(path) if path else None, "keys": [], "shape": None, "dtype": None, "error": None}
    if path is None:
        result["error"] = "Artifact not available"
        return result
    try:
        import numpy as np

        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as values:
                result["keys"] = list(values.files)
                preferred = (
                    "data",
                    "cube",
                    "fused",
                    "prediction",
                    "spatial_prediction",
                    "bicubic",
                    "original",
                    "U",
                    "S",
                    "sam_map",
                )
                key = next((name for name in preferred if name in values.files), None)
                if key is None:
                    result["error"] = "No recognized array key"
                else:
                    result["shape"] = tuple(values[key].shape)
                    result["dtype"] = str(values[key].dtype)
        else:
            values = np.load(path, allow_pickle=False)
            result["shape"] = tuple(values.shape)
            result["dtype"] = str(values.dtype)
    except (OSError, ValueError, KeyError) as error:
        result["error"] = str(error)
    return result


def artifact_state() -> str:
    present = [artifact(role) is not None for role in ("input", "bicubic", "sen2sr", "spectraguard")]
    if all(present):
        return "Results Ready"
    if any(present):
        return "Incomplete Results"
    return "Processing Required"


def _metadata() -> dict:
    path = artifact("metadata")
    if path is None:
        return {}
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        return values if isinstance(values, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _metrics() -> dict:
    path = artifact("metrics")
    if path is None:
        return {}
    results = {}
    if path.suffix == ".npz" or (path.suffix == ".json" and (path.parent / "metrics.npz").is_file()):
        npz_path = path if path.suffix == ".npz" else path.parent / "metrics.npz"
        try:
            import numpy as np

            with np.load(npz_path, allow_pickle=False) as values:
                for key in values.files:
                    if key == "sam_map":
                        continue
                    val = values[key]
                    if isinstance(val, np.ndarray):
                        if val.ndim == 0:
                            results[key] = val.item()
                        elif val.dtype.kind in ("M", "m", "O", "U", "S"):
                            results[key] = val.tolist()
                        elif val.size < 50:
                            results[key] = val.tolist()
                    else:
                        results[key] = val
        except (OSError, ValueError):
            pass
    if path.suffix == ".json" and path.is_file():
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(values, dict):
                results.update(values)
        except (OSError, json.JSONDecodeError):
            pass
    return results


def model_metrics(role: str) -> dict:
    if role == "spectraguard":
        m = _metrics()
        if m:
            return m
    path = artifact(role)
    input_path = artifact("input")
    if path is None or input_path is None:
        return {}
    try:
        import numpy as np
        from src.validation.metrics import compute_sam

        inp, bands_inp = _npz_data("input")
        data, bands = _npz_data(role)

        h, w = inp.shape[1], inp.shape[2]
        h_d, w_d = data.shape[1], data.shape[2]
        sy, sx = max(1, h_d // h), max(1, w_d // w)
        down = data[:, :h * sy, :w * sx].reshape(data.shape[0], h, sy, w, sx).mean(axis=(2, 4))

        if bands and bands != bands_inp:
            common = [b for b in bands_inp if b in bands]
            idx_inp = [bands_inp.index(b) for b in common]
            idx_d = [bands.index(b) for b in common]
            sub_d = down[idx_d]
            sub_inp = inp[idx_inp]
            use_bands = common
        else:
            sub_d = down
            sub_inp = inp[:down.shape[0]]
            use_bands = bands_inp[:down.shape[0]]

        sam_map = compute_sam(sub_d, sub_inp)
        sam_mean = float(np.nanmean(sam_map))
        diff = sub_d - sub_inp
        rmse = {b: float(np.sqrt(np.mean(diff[i] ** 2))) for i, b in enumerate(use_bands)}
        mae = {b: float(np.mean(np.abs(diff[i]))) for i, b in enumerate(use_bands)}

        return {
            "sam_mean": sam_mean,
            "per_band_rmse": rmse,
            "per_band_mae": mae,
            "reference_type": "internal_consistency",
        }
    except Exception:
        return {}


def _endmember_matrix() -> tuple[list[str], list[str], list[list[float]]] | None:
    try:
        from src.library.s2c_usgs_library import load_active_library, prepare_endmembers

        library, response = load_active_library(str(ROOT), str(ENDMEMBER_MANIFEST), str(SENSOR_RESPONSE))
        endmembers = prepare_endmembers(library, response)
        return list(response.band_names), list(endmembers.names), endmembers.A.tolist()
    except (OSError, KeyError, TypeError, ValueError, ImportError):
        return None


def _npz_data(role: str) -> tuple[object, list[str]]:
    path = artifact(role)
    if path is None:
        raise FileNotFoundError(f"{role} artifact is not available")
    import numpy as np

    with np.load(path, allow_pickle=False) as values:
        preferred = ("data", "cube", "fused", "prediction", "spatial_prediction", "bicubic", "original")
        key = next((candidate for candidate in preferred if candidate in values.files), None)
        if key is None:
            raise ValueError(f"{role} artifact has no recognized image array")
        array = np.asarray(values[key])
        band_key = "source_bands" if role == "sen2sr" else "band_names"
        bands = [str(value) for value in values[band_key].tolist()] if band_key in values.files else []
        return array, bands


def sample_pixel_spectrum(rel_x: float, rel_y: float) -> dict:
    import numpy as np

    result = {
        "x": rel_x,
        "y": rel_y,
        "bands": ["B02", "B03", "B04", "B08", "B11"],
        "wavelengths": [490, 560, 665, 842, 1610],
        "curves": {},
    }

    for role in ("input", "bicubic", "sen2sr", "spectraguard"):
        path = artifact(role)
        if path is None:
            continue
        try:
            data, bands = _npz_data(role)
            array = np.asarray(data, dtype=np.float32)
            if array.ndim == 3:
                h, w = array.shape[1], array.shape[2]
                px = max(0, min(int(rel_x * w), w - 1))
                py = max(0, min(int(rel_y * h), h - 1))

                curve = []
                for band_name in ("B02", "B03", "B04", "B08", "B11"):
                    if bands and band_name in bands:
                        idx = bands.index(band_name)
                        curve.append(float(array[idx, py, px]))
                    elif band_name in ("B02", "B03", "B04", "B08") and array.shape[0] >= 4:
                        idx = ("B02", "B03", "B04", "B08").index(band_name)
                        curve.append(float(array[idx, py, px]))
                    else:
                        curve.append(None)
                result["curves"][role] = curve
        except Exception:
            pass

    return result


def _stretch(array):
    import numpy as np

    valid = array[np.isfinite(array) & (array > 0)]
    if not valid.size:
        return np.zeros(array.shape, dtype=np.uint8)
    sub = valid[::4] if valid.size > 10000 else valid
    low, high = np.percentile(sub, (2, 98))
    span = max(float(high - low), 1e-6)
    scaled = (array - low) * 255.0 / span
    return np.clip(scaled, 0, 255).astype(np.uint8)


def as_png(rgb) -> bytes:
    try:
        from PIL import Image

        image = Image.fromarray(rgb)
        max_dim = max(image.width, image.height)
        if max_dim > 1024:
            scale = 1024.0 / max_dim
            new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(new_size, Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()
    except ImportError:
        height, width, _ = rgb.shape
        raw = b"".join(b"\0" + rgb[y].tobytes() for y in range(height))

        def chunk(name, body):
            return struct.pack(">I", len(body)) + name + body + struct.pack(">I", zlib.crc32(name + body) & 0xFFFFFFFF)

        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, level=6))
            + chunk(b"IEND", b"")
        )


def _scientific_colormap(normalized) -> object:
    import numpy as np

    v = np.clip(normalized, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(v * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(v * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(v * 4.0 - 1.0), 0.0, 1.0)
    r = (0.05 + 0.95 * r) * 255.0
    g = (0.05 + 0.95 * g) * 255.0
    b = (0.15 + 0.85 * b) * 255.0
    return np.dstack([r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)])


def _rgb_preview(role: str, mode: str = "rgb") -> bytes:
    path = artifact(role)
    if path is None:
        raise FileNotFoundError(f"{role} artifact unavailable")
    mtime = path.stat().st_mtime
    cache_key = (f"rgb:{role}:{mode}", mtime)
    if cache_key in _IMAGE_CACHE:
        return _IMAGE_CACHE[cache_key]

    import numpy as np

    data, bands = _npz_data(role)
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 2:
        channels = [array] * 3
    elif array.ndim == 3:
        if bands:
            if mode == "nir" and all(b in bands for b in ("B08", "B04", "B03")):
                channels = [array[bands.index(b)] for b in ("B08", "B04", "B03")]
            elif mode == "swir" and all(b in bands for b in ("B11", "B08", "B04")):
                channels = [array[bands.index(b)] for b in ("B11", "B08", "B04")]
            elif all(b in bands for b in ("B04", "B03", "B02")):
                channels = [array[bands.index(b)] for b in ("B04", "B03", "B02")]
            else:
                channels = list(array[:3])
        elif array.shape[0] <= 32:
            channels = list(array[:3])
        else:
            channels = list(array[..., :3].transpose(2, 0, 1))
        channels = (channels + [channels[-1]] * 3)[:3]
    else:
        raise ValueError("Artifact image array has unsupported dimensions")

    png_bytes = as_png(np.dstack([_stretch(channel) for channel in channels]))
    _IMAGE_CACHE[cache_key] = png_bytes
    return png_bytes


def _map_preview(role: str) -> bytes:
    path = artifact(role)
    if path is None:
        raise FileNotFoundError(f"{role} artifact is not available")
    mtime = path.stat().st_mtime
    cache_key = (f"map:{role}", mtime)
    if cache_key in _IMAGE_CACHE:
        return _IMAGE_CACHE[cache_key]

    import numpy as np

    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as values:
            if role == "sam_map":
                key = next((k for k in ("sam_map", "sam") if k in values.files), None)
            else:
                key = next((k for k in ("U", "uncertainty", "sam") if k in values.files), None)
            if key is None:
                raise ValueError(f"{role} artifact has no displayable map")
            array = values[key]
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{role} map has shape {array.shape}; expected a 2-D map")

    valid = array[np.isfinite(array)]
    if valid.size > 0:
        low, high = np.percentile(valid[::2] if valid.size > 10000 else valid, (2, 98))
        norm = (array - low) / max(float(high - low), 1e-6)
    else:
        norm = np.zeros_like(array)

    png_bytes = as_png(_scientific_colormap(norm))
    _IMAGE_CACHE[cache_key] = png_bytes
    return png_bytes


def output_plot_bytes(plot_name: str) -> bytes:
    file_path = OUTPUTS / f"{plot_name}.png"
    if not file_path.is_file():
        raise FileNotFoundError(f"Generated plot {plot_name}.png unavailable")
    return file_path.read_bytes()


def _format_time(raw: str) -> str:
    if not raw:
        return "2026-08-31 05:26:41 UTC"
    try:
        if "T" in str(raw):
            date_part, time_part = str(raw).split("T", 1)
            time_clean = time_part.split(".")[0].split("+")[0].split("Z")[0]
            return f"{date_part} {time_clean} UTC"
    except Exception:
        pass
    return str(raw)


def _format_bounds(bounds: object) -> str:
    if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        minx, miny, maxx, maxy = bounds
        return f"X: {minx:g}–{maxx:g}<br>Y: {miny:g}–{maxy:g}"
    if isinstance(bounds, dict):
        minx = bounds.get("minx", "N/A")
        miny = bounds.get("miny", "N/A")
        maxx = bounds.get("maxx", "N/A")
        maxy = bounds.get("maxy", "N/A")
        return f"X: {minx}–{maxx}<br>Y: {miny}–{maxy}"
    return str(bounds) if bounds else "Configured"


def _scene_info(data: dict) -> dict:
    meta = _metadata()
    scene_name = meta.get("scene_id") or get_config(data, "data", "scene_path", default="")
    scene_path_obj = Path(scene_name)
    if not scene_path_obj.is_absolute():
        scene_path_obj = ROOT / scene_path_obj

    info = {
        "scene": scene_path_obj.name or "Sentinel-2 Scene",
        "crs": meta.get("aoi", {}).get("crs") or "EPSG:32643 (UTM 43N)",
        "resolution": get_config(data, "data", "working_resolution_m"),
        "acquisition": _format_time(meta.get("timestamp") or "2026-08-31T05:26:41Z"),
        "bounds": meta.get("aoi", {}).get("bounds"),
    }
    try:
        import rasterio

        band = next(scene_path_obj.rglob("*_B02_10m.jp2"))
        with rasterio.open(band) as source:
            info["crs"] = source.crs.to_string() if source.crs else info["crs"]
            info["resolution"] = f"{abs(source.transform.a):g} m"
            info["bounds"] = [source.bounds.left, source.bounds.bottom, source.bounds.right, source.bounds.top]
    except (OSError, StopIteration, ImportError):
        input_art = artifact("input")
        if input_art:
            try:
                import numpy as np

                with np.load(input_art, allow_pickle=False) as values:
                    if "crs" in values.files:
                        info["crs"] = str(values["crs"])
                    if "bounds" in values.files and not info["bounds"]:
                        info["bounds"] = values["bounds"].tolist()
                    if "acquisition_time" in values.files and not meta.get("timestamp"):
                        info["acquisition"] = _format_time(str(values["acquisition_time"]))
            except (OSError, ValueError):
                pass
    return info


def scene_bands(data: dict) -> list[Path] | None:
    scene_value = get_config(data, "data", "scene_path", default="")
    scene = Path(scene_value)
    if not scene.is_absolute():
        scene = ROOT / scene
    if not scene.is_dir():
        return None
    paths = []
    for band in ("B04", "B03", "B02"):
        matches = sorted(scene.rglob(f"*_{band}_10m.jp2"))
        if not matches:
            return None
        paths.append(matches[0])
    return paths


def raster_preview(data: dict) -> bytes:
    cache_key = ("scene_preview", 0.0)
    if cache_key in _IMAGE_CACHE:
        return _IMAGE_CACHE[cache_key]

    import numpy as np

    paths = scene_bands(data)
    if paths is not None:
        try:
            import rasterio

            arrays = []
            for path in paths:
                with rasterio.open(path) as source:
                    arrays.append(
                        source.read(1, out_shape=(1024, 1024), masked=True).filled(0).astype(np.float32) / 10000.0
                    )
            png_bytes = as_png(np.dstack([_stretch(arr) for arr in arrays]))
            _IMAGE_CACHE[cache_key] = png_bytes
            return png_bytes
        except (OSError, ImportError):
            pass

    input_art = artifact("input") or artifact("bicubic")
    if input_art:
        png_bytes = _rgb_preview("input" if artifact("input") else "bicubic")
        _IMAGE_CACHE[cache_key] = png_bytes
        return png_bytes

    raise FileNotFoundError("Input scene imagery unavailable")


def cards(data: dict) -> str:
    meta = _metadata()
    model_name = meta.get("model_name") or get_config(data, "sr", "model")
    scale_factor = meta.get("scale") or get_config(data, "sr", "scale")
    fields = (
        ("Input resolution", get_config(data, "data", "working_resolution_m"), ICONS["layer"]),
        ("Target resolution", get_config(data, "data", "target_resolution_m"), ICONS["globe"]),
        ("Input bands", get_config(data, "data", "bands"), ICONS["chart"]),
        ("SR model", str(model_name), ICONS["shield"]),
        ("Scale factor", f"{scale_factor}×", ICONS["expand"]),
        ("Compute device", get_config(data, "sr", "device").upper(), ICONS["process"]),
        ("Sampling steps", get_config(data, "sr", "sampling_steps"), ICONS["analysis"]),
        ("Fixed AOI", "Configured (Fixed)", ICONS["scene"]),
    )
    return (
        '<div class="grid-cards">'
        + "".join(
            f'<div class="metric-card">'
            f'<div class="metric-icon">{icon}</div>'
            f'<div class="metric-content">'
            f'<small>{html.escape(name)}</small>'
            f'<strong>{html.escape(value)}</strong>'
            f'</div>'
            f'</div>'
            for name, value, icon in fields
        )
        + "</div>"
    )


def _aoi_overlay(data: dict, info: dict) -> str:
    aoi = data.get("data", {}).get("aoi")
    bounds = info.get("bounds")
    if not isinstance(aoi, dict) or not bounds:
        return ""
    left, bottom, right, top = bounds
    x = max(0, (aoi["minx"] - left) / (right - left) * 100)
    y = max(0, (top - aoi["maxy"]) / (top - bottom) * 100)
    width = max(0, (aoi["maxx"] - aoi["minx"]) / (right - left) * 100)
    height = max(0, (aoi["maxy"] - aoi["miny"]) / (top - bottom) * 100)
    return (
        f'<div class="aoi-box" style="left:{x:.3f}%;top:{y:.3f}%;width:{width:.3f}%;height:{height:.3f}%">'
        f'<div class="aoi-corner top-left"></div><div class="aoi-corner top-right"></div>'
        f'<div class="aoi-corner bottom-left"></div><div class="aoi-corner bottom-right"></div>'
        f'<span>🎯 Fixed Processing AOI</span>'
        f'</div>'
    )


def viewer(src: str, unavailable: str, viewer_id: str = "interactive-viewer", img_id: str = "scientific-image", extra_overlay: str = "") -> str:
    return (
        f'<div class="viewer-container">'
        f'<div class="hud-toolbar">'
        f'<div class="hud-left">'
        f'<span class="cursor-coords-badge hud-badge-mono">Cursor: X: --% Y: --%</span>'
        f'</div>'
        f'<div class="hud-right">'
        f'<span class="zoom-badge hud-badge">100%</span>'
        f'<button class="hud-btn" onclick="zoom(-.2)" title="Zoom Out">−</button>'
        f'<button class="hud-btn" onclick="zoom(.2)" title="Zoom In">+</button>'
        f'<button class="hud-btn" onclick="resetView()">Reset View</button>'
        f'<button class="hud-btn highlight" onclick="openLightbox(\'{src}\')">{ICONS["expand"]} Expand</button>'
        f'</div>'
        f'</div>'
        f'<div class="viewer" id="{viewer_id}">'
        f'<img id="{img_id}" class="interactive-image" src="{src}" alt="Scientific raster preview" '
        f'onerror="this.remove();this.parentElement.innerHTML=\'<div class=empty>{unavailable}</div>\'">'
        f'{extra_overlay}'
        f'</div>'
        f'</div>'
    )



CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-dark: #070a12;
    --bg-dark-gradient: radial-gradient(circle at 50% 0%, #0f192e 0%, #070a12 75%);
    --bg-sidebar: rgba(11, 17, 30, 0.88);
    --bg-topbar: rgba(11, 17, 30, 0.8);
    --bg-card: rgba(16, 24, 42, 0.75);
    --bg-card-hover: rgba(22, 34, 58, 0.9);
    --bg-viewer: #03060a;
    --border-color: rgba(255, 255, 255, 0.08);
    --border-bright: rgba(6, 182, 212, 0.4);
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
    --accent-cyan: #06b6d4;
    --accent-cyan-bright: #00f2fe;
    --accent-cyan-glow: rgba(6, 182, 212, 0.22);
    --accent-teal: #10b981;
    --accent-gold: #f59e0b;
    --accent-purple: #8b5cf6;
    --shadow-card: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
    --shadow-glow: 0 0 25px rgba(6, 182, 212, 0.25);
    --code-font: 'JetBrains Mono', monospace;
    --heading-font: 'Outfit', sans-serif;
    --body-font: 'Inter', sans-serif;
}

[data-theme="light"] {
    --bg-dark: #f8fafc;
    --bg-dark-gradient: linear-gradient(180deg, #edf2f7 0%, #f8fafc 100%);
    --bg-sidebar: rgba(255, 255, 255, 0.92);
    --bg-topbar: rgba(255, 255, 255, 0.88);
    --bg-card: #ffffff;
    --bg-card-hover: #f1f5f9;
    --bg-viewer: #0f172a;
    --border-color: #e2e8f0;
    --border-bright: #cbd5e1;
    --text-primary: #0f172a;
    --text-muted: #64748b;
    --accent-cyan: #0284c7;
    --accent-cyan-bright: #0369a1;
    --accent-cyan-glow: rgba(2, 132, 199, 0.12);
    --accent-teal: #059669;
    --accent-gold: #d97706;
    --accent-purple: #7c3aed;
    --shadow-card: 0 4px 20px -2px rgba(0, 0, 0, 0.06);
    --shadow-glow: 0 0 15px rgba(2, 132, 199, 0.15);
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: var(--bg-dark);
    background-image: var(--bg-dark-gradient);
    color: var(--text-primary);
    font-family: var(--body-font);
    font-size: 14px;
    line-height: 1.6;
    transition: background 0.3s ease, color 0.3s ease;
    min-height: 100vh;
    overflow-x: hidden;
}

/* Layout Grid */
.layout { display: flex; min-height: 100vh; }
.side {
    width: 270px;
    padding: 28px 18px;
    background: var(--bg-sidebar);
    backdrop-filter: blur(16px);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    z-index: 100;
    position: sticky;
    top: 0;
    height: 100vh;
}

.brand {
    font-family: var(--heading-font);
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.6px;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 8px 12px;
}
.brand b { color: var(--accent-cyan-bright); }
.brand-badge {
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    padding: 3px 8px;
    background: var(--accent-cyan-glow);
    color: var(--accent-cyan-bright);
    border-radius: 6px;
    border: 1px solid var(--accent-cyan);
    letter-spacing: 0.5px;
}
.sub {
    font-size: 10.5px;
    color: var(--text-muted);
    margin: -6px 8px 28px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}

/* Navigation Links */
.nav { display: flex; flex-direction: column; gap: 6px; flex: 1; }
.nav a {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    color: var(--text-muted);
    text-decoration: none;
    border-radius: 10px;
    font-weight: 500;
    font-size: 14px;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    border: 1px solid transparent;
}
.nav a svg { color: var(--text-muted); transition: color 0.2s ease, transform 0.2s ease; }
.nav a:hover {
    background: var(--bg-card-hover);
    color: var(--text-primary);
    border-color: var(--border-color);
    transform: translateX(3px);
}
.nav a:hover svg { color: var(--accent-cyan-bright); transform: scale(1.1); }
.nav a.active {
    background: var(--accent-cyan-glow);
    color: var(--accent-cyan-bright);
    font-weight: 700;
    border: 1px solid var(--accent-cyan);
    box-shadow: 0 4px 15px rgba(6, 182, 212, 0.15);
}
.nav a.active svg { color: var(--accent-cyan-bright); }

/* Sidebar Footer */
.side-footer {
    padding: 14px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    margin-top: auto;
    font-size: 11.5px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.pulse-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-teal);
    box-shadow: 0 0 10px var(--accent-teal);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Topbar Header */
.main-wrapper { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 44px;
    background: var(--bg-topbar);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border-color);
    position: sticky;
    top: 0;
    z-index: 90;
}
.topbar-info { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.topbar-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 14px;
    border-radius: 20px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    color: var(--text-muted);
    font-weight: 600;
    font-family: var(--code-font);
    font-size: 11.5px;
    max-width: 420px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.topbar-pill strong { color: var(--accent-cyan-bright); }

.theme-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    border: 1px solid var(--border-color);
    background: var(--bg-card);
    color: var(--text-primary);
    padding: 7px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
}
.theme-btn:hover { background: var(--bg-card-hover); border-color: var(--accent-cyan); transform: scale(1.02); }

/* Main Content Area */
.main { max-width: 1550px; padding: 40px 44px 80px; width: 100%; margin: 0 auto; }
.app-view { display: none; opacity: 1; }
.app-view.active { display: block !important; opacity: 1 !important; }

.eyebrow {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent-cyan-bright);
    margin-bottom: 6px;
}
h1 { font-family: var(--heading-font); font-size: 34px; font-weight: 800; margin-bottom: 10px; letter-spacing: -0.8px; color: var(--text-primary); }
h2 { font-family: var(--heading-font); font-size: 22px; font-weight: 700; margin: 40px 0 18px; color: var(--text-primary); display: flex; align-items: center; gap: 10px; }
.muted { color: var(--text-muted); line-height: 1.6; font-size: 14.5px; }

/* Status Badges */
.status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 20px;
    padding: 6px 16px;
    color: var(--accent-teal);
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    font-size: 12px;
    font-weight: 700;
}
.status::before {
    content: "";
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-teal);
    box-shadow: 0 0 10px var(--accent-teal);
}
.status.warn {
    color: var(--accent-gold);
    background: rgba(245, 158, 11, 0.12);
    border-color: rgba(245, 158, 11, 0.35);
}
.status.warn::before { background: var(--accent-gold); box-shadow: 0 0 10px var(--accent-gold); }

/* Metric Cards Grid */
.grid-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin: 28px 0;
}
.metric-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 20px;
    box-shadow: var(--shadow-card);
    display: flex;
    align-items: flex-start;
    gap: 14px;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    min-width: 0;
}
.metric-card:hover {
    border-color: var(--border-bright);
    transform: translateY(-3px);
    box-shadow: var(--shadow-glow);
}
.metric-icon {
    width: 40px; height: 40px;
    border-radius: 10px;
    background: var(--accent-cyan-glow);
    color: var(--accent-cyan-bright);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.metric-content { min-width: 0; flex: 1; }
.metric-content small { display: block; color: var(--text-muted); text-transform: uppercase; font-size: 10px; font-weight: 800; letter-spacing: 1.2px; }
.metric-content strong {
    display: block;
    margin-top: 6px;
    font-size: 14.5px;
    color: var(--text-primary);
    font-family: var(--code-font);
    font-weight: 600;
    word-break: break-all;
    overflow-wrap: anywhere;
    line-height: 1.4;
}

/* Interactive Swipe Curtain Slider */
.swipe-container {
    position: relative;
    width: 100%;
    height: 560px;
    overflow: hidden;
    border-radius: 16px;
    border: 1px solid var(--border-color);
    background: var(--bg-viewer);
    box-shadow: var(--shadow-card);
    user-select: none;
    margin: 20px 0 32px;
}
.swipe-after {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: cover;
}
.swipe-before-wrapper {
    position: absolute;
    top: 0; left: 0;
    width: 50%; height: 100%;
    overflow: hidden;
    border-right: 3px solid var(--accent-cyan-bright);
    box-shadow: 6px 0 25px rgba(0, 242, 254, 0.5);
}
.swipe-before {
    position: absolute;
    top: 0; left: 0;
    object-fit: cover;
}
.swipe-handle {
    position: absolute;
    top: 0; left: 50%;
    height: 100%;
    transform: translateX(-50%);
    cursor: ew-resize;
    z-index: 20;
    display: flex;
    align-items: center;
    justify-content: center;
}
.swipe-circle {
    width: 44px; height: 44px;
    border-radius: 50%;
    background: var(--accent-cyan-bright);
    color: #070a12;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 900;
    box-shadow: 0 0 25px var(--accent-cyan-bright);
    border: 3px solid #ffffff;
}
.swipe-badge-left, .swipe-badge-right {
    position: absolute;
    top: 20px;
    padding: 8px 18px;
    background: rgba(7, 10, 18, 0.88);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    font-size: 12px;
    font-weight: 800;
    color: var(--accent-cyan-bright);
    font-family: var(--code-font);
    z-index: 10;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}
.swipe-badge-left { left: 20px; }
.swipe-badge-right { right: 20px; color: var(--accent-teal); }

/* Plot Gallery Card */
.plot-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 28px;
    box-shadow: var(--shadow-card);
}
.plot-card h3 {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.plot-card img {
    width: 100%;
    max-height: 560px;
    object-fit: contain;
    background: var(--bg-viewer);
    border-radius: 10px;
    border: 1px solid var(--border-color);
    display: block;
    cursor: zoom-in;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.plot-card img:hover { border-color: var(--accent-cyan-bright); }

/* Spectral Profile Inspector Panel */
.spectral-panel {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-bright);
    border-radius: 14px;
    padding: 24px;
    margin: 28px 0;
    box-shadow: var(--shadow-glow);
}
.spectral-chart-wrapper {
    height: 270px;
    width: 100%;
    position: relative;
    margin-top: 16px;
}

/* Modal Lightbox */
.lightbox {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(3, 6, 12, 0.94);
    backdrop-filter: blur(12px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 2000;
    padding: 40px;
}
.lightbox.active { display: flex !important; }
.lightbox img {
    max-width: 95vw;
    max-height: 90vh;
    object-fit: contain;
    border-radius: 12px;
    border: 1px solid var(--border-bright);
    box-shadow: 0 25px 60px rgba(0,0,0,0.9);
}
.lightbox-close {
    position: absolute;
    top: 24px; right: 32px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    font-size: 24px;
    line-height: 1;
    padding: 10px 16px;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.lightbox-close:hover { background: #ef4444; color: #fff; border-color: #ef4444; }

/* Interactive Canvas Viewer HUD */
.viewer-container {
    position: relative;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-card);
    background: var(--bg-viewer);
}
.viewer {
    min-height: 540px;
    overflow: hidden;
    position: relative;
    touch-action: none;
}
.viewer img {
    width: 100%;
    height: 540px;
    object-fit: contain;
    display: block;
    transform-origin: center;
    cursor: crosshair;
}
.hud-toolbar {
    position: absolute;
    top: 16px; left: 16px; right: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(11, 17, 30, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 6px 12px;
    z-index: 10;
}
.hud-left, .hud-right { display: flex; align-items: center; gap: 8px; }
.hud-btn {
    border: 1px solid var(--border-color);
    background: var(--bg-card);
    color: var(--text-primary);
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 12px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.15s ease;
}
.hud-btn:hover { background: var(--bg-card-hover); color: var(--accent-cyan-bright); border-color: var(--accent-cyan); }
.hud-btn.highlight { background: var(--accent-cyan-glow); color: var(--accent-cyan-bright); border-color: var(--accent-cyan); }
.hud-badge {
    font-family: var(--code-font);
    font-size: 11px;
    color: var(--accent-cyan-bright);
    padding: 4px 10px;
    background: var(--accent-cyan-glow);
    border-radius: 6px;
    border: 1px solid var(--accent-cyan);
}
.hud-badge-mono {
    font-family: var(--code-font);
    font-size: 11.5px;
    color: var(--text-muted);
}

/* Tabs & Action Buttons */
.buttons { margin: 20px 0; display: flex; flex-wrap: wrap; gap: 10px; }
.buttons a, .button, .tab {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid var(--border-color);
    border-radius: 10px;
    background: var(--bg-card);
    color: var(--text-primary);
    text-decoration: none;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 13.5px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.buttons a:hover, .button:hover, .tab:hover {
    background: var(--bg-card-hover);
    color: var(--text-primary);
    border-color: var(--accent-cyan);
    transform: translateY(-1px);
}
.tab.active {
    background: var(--accent-cyan);
    color: #ffffff;
    border-color: var(--accent-cyan-bright);
    font-weight: 700;
    box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3);
}

/* Workflow Step Matrix */
.workflow { max-width: 900px; margin: 28px 0; }
.step {
    padding: 18px 24px;
    margin: 12px 0;
    border-left: 4px solid var(--accent-cyan);
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: var(--shadow-card);
    border-top: 1px solid var(--border-color);
    border-right: 1px solid var(--border-color);
    border-bottom: 1px solid var(--border-color);
}
.step.complete { border-left-color: var(--accent-teal); }
.step.missing { border-left-color: var(--text-muted); opacity: 0.7; }
.step-title { font-size: 15px; font-weight: 600; color: var(--text-primary); }
.step-status { font-weight: 800; font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; padding: 4px 10px; border-radius: 6px; }
.step.complete .step-status { color: var(--accent-teal); background: rgba(16, 185, 129, 0.12); }
.step.missing .step-status { color: var(--text-muted); background: rgba(148, 163, 184, 0.12); }
.arrow { text-align: center; color: var(--accent-cyan); font-weight: bold; font-size: 16px; margin: 4px 0; opacity: 0.6; }

/* AOI Scene Map */
.aoi-scene { position: relative; border-radius: 14px; overflow: hidden; border: 1px solid var(--border-color); box-shadow: var(--shadow-card); }
.aoi-scene img { width: 100%; height: 540px; object-fit: contain; display: block; background: var(--bg-viewer); }
.aoi-box {
    position: absolute;
    border: 2px solid var(--accent-gold);
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.5);
    pointer-events: none;
}
.aoi-box span {
    position: absolute;
    top: 8px; left: 8px;
    white-space: nowrap;
    background: rgba(7, 10, 18, 0.92);
    backdrop-filter: blur(6px);
    border: 1px solid var(--accent-gold);
    padding: 5px 12px;
    color: var(--accent-gold);
    font-size: 11.5px;
    font-weight: 800;
    border-radius: 6px;
    font-family: var(--code-font);
}

/* Comparison Grids & Tables */
.comparison {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    margin: 28px 0;
}
.comparison .card { padding: 16px; display: flex; flex-direction: column; gap: 12px; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; }
.comparison img, .map-image {
    width: 100%;
    height: 240px;
    object-fit: contain;
    background: var(--bg-viewer);
    border-radius: 8px;
    border: 1px solid var(--border-color);
    cursor: zoom-in;
    transform: none !important;
}
.chip {
    display: inline-block;
    padding: 4px 10px;
    background: var(--accent-cyan-glow);
    border: 1px solid var(--border-bright);
    color: var(--accent-cyan-bright);
    border-radius: 6px;
    font-size: 11.5px;
    font-family: var(--code-font);
    margin: 2px 4px 2px 0;
}

.table { width: 100%; border-collapse: collapse; margin: 24px 0; border-radius: 12px; overflow: hidden; border: 1px solid var(--border-color); }
.table th, .table td { text-align: left; padding: 16px 20px; border-bottom: 1px solid var(--border-color); }
.table th { color: var(--text-muted); font-size: 11.5px; text-transform: uppercase; letter-spacing: 1.2px; background: var(--bg-sidebar); font-weight: 800; }
.table td { background: var(--bg-card); font-family: var(--code-font); font-size: 13.5px; }
.table tr:hover td { background: var(--bg-card-hover); }
.good { color: var(--accent-teal); font-weight: 700; }
.warn { color: var(--accent-gold); font-weight: 700; }
.fixed { color: var(--accent-gold); font-weight: 600; margin: 8px 0 16px; }

@media (max-width: 1100px) { .comparison { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 768px) {
    .layout { flex-direction: column; }
    .side { width: 100%; height: auto; position: relative; border-right: none; border-bottom: 1px solid var(--border-color); }
    .main { padding: 24px 20px; }
    .topbar { padding: 16px 20px; }
    .comparison { grid-template-columns: 1fr; }
}
"""

SCRIPT = """<script>
let viewState = { scale: 1, x: 0, y: 0 };
let currentRole = 'spectraguard';
let currentMode = 'rgb';

function updateClock() {
    let now = new Date();
    let clock = document.getElementById('utc-clock');
    if (clock) clock.innerText = now.toISOString().substring(11, 19) + ' UTC';
}

function normalizeRoute(path) {
    if (!path || path === '') return '/';
    if (path.length > 1 && path.endsWith('/')) return path.slice(0, -1);
    return path;
}

function showRouteView(route) {
    try {
        let cleanRoute = normalizeRoute(route);
        let routeMap = {
            '/': 'view-home',
            '/scene': 'view-scene',
            '/process': 'view-process',
            '/results': 'view-results',
            '/analysis': 'view-analysis',
            '/validation': 'view-validation',
            '/fcls': 'view-fcls',
            '/about': 'view-about'
        };
        let targetId = routeMap[cleanRoute] || 'view-home';

        document.querySelectorAll('.app-view').forEach(el => {
            el.classList.remove('active');
            el.style.display = 'none';
        });

        let targetView = document.getElementById(targetId);
        if (targetView) {
            targetView.classList.add('active');
            targetView.style.display = 'block';
        } else {
            let homeView = document.getElementById('view-home');
            if (homeView) {
                homeView.classList.add('active');
                homeView.style.display = 'block';
            }
        }

        document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));
        let activeNav = document.querySelector(`.nav-link[data-route="${cleanRoute}"]`);
        if (activeNav) activeNav.classList.add('active');

        try { resetView(); } catch(e) {}

        if (cleanRoute === '/results' || cleanRoute === '/scene') {
            setTimeout(() => {
                try { initSwipe(); } catch(e) {}
            }, 50);
        }
    } catch (err) {
        console.error('Navigation error:', err);
    }
}

function navigateTo(route, e) {
    if (e) e.preventDefault();
    try {
        history.pushState({ route: route }, '', route);
    } catch(err) {}
    showRouteView(route);
}

function getActiveImage() {
    try {
        let activeView = document.querySelector('.app-view.active');
        if (!activeView) return null;
        return activeView.querySelector('.viewer img.interactive-image') || activeView.querySelector('#scientific-image') || activeView.querySelector('#scene-image') || activeView.querySelector('.viewer img');
    } catch(e) {
        return null;
    }
}

function paint() {
    try {
        let img = getActiveImage();
        if (img) img.style.transform = `translate(${viewState.x}px, ${viewState.y}px) scale(${viewState.scale})`;
        let activeView = document.querySelector('.app-view.active');
        if (activeView) {
            let badge = activeView.querySelector('.zoom-badge');
            if (badge) badge.innerText = Math.round(viewState.scale * 100) + '%';
        }
    } catch(e) {}
}

function zoom(delta) {
    viewState.scale = Math.min(8, Math.max(0.35, viewState.scale + delta));
    paint();
}

function resetView() {
    viewState = { scale: 1, x: 0, y: 0 };
    paint();
}

function updateViewerSrc() {
    let img = document.querySelector('#scientific-image');
    if (img) {
        img.src = '/result/' + currentRole + '_' + currentMode + '.png';
        resetView();
    }
    let note = document.getElementById('sen2sr-swir-note');
    if (note) {
        if (currentRole === 'sen2sr' && currentMode === 'swir') {
            note.style.display = 'block';
        } else {
            note.style.display = 'none';
        }
    }
}

function selectResult(role, button) {
    document.querySelectorAll('.tab-product').forEach(x => x.classList.remove('active'));
    button.classList.add('active');
    currentRole = role;
    updateViewerSrc();
}

function selectMode(mode, button) {
    document.querySelectorAll('.tab-mode').forEach(x => x.classList.remove('active'));
    button.classList.add('active');
    currentMode = mode;
    updateViewerSrc();
}

function toggleTheme() {
    let current = document.documentElement.getAttribute('data-theme') || 'dark';
    let next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('sg_theme', next);
}

function openLightbox(src) {
    let box = document.querySelector('#lightbox-modal');
    let img = document.querySelector('#lightbox-img');
    if (box && img) {
        img.src = src;
        box.classList.add('active');
    }
}

function closeLightbox() {
    let box = document.querySelector('#lightbox-modal');
    if (box) box.classList.remove('active');
}

// Swipe Comparison Slider
function initSwipe() {
    let box = document.getElementById('swipe-box');
    let wrap = document.getElementById('swipe-before-wrap');
    let handle = document.getElementById('swipe-handle');
    let beforeImg = document.querySelector('.swipe-before');
    if (!box || !wrap || !handle) return;

    function syncDimensions() {
        let rect = box.getBoundingClientRect();
        if (rect.width > 0 && beforeImg) {
            beforeImg.style.width = rect.width + 'px';
            beforeImg.style.height = rect.height + 'px';
        }
    }

    let dragging = false;
    function move(clientX) {
        let rect = box.getBoundingClientRect();
        if (rect.width === 0) return;
        syncDimensions();
        let pos = Math.max(0, Math.min(clientX - rect.left, rect.width));
        let pct = (pos / rect.width) * 100;
        wrap.style.width = pct + '%';
        handle.style.left = pct + '%';
    }

    syncDimensions();
    window.addEventListener('resize', syncDimensions);

    box.onpointerdown = e => { dragging = true; move(e.clientX); box.setPointerCapture(e.pointerId); };
    box.onpointermove = e => { if (dragging) move(e.clientX); };
    box.onpointerup = () => { dragging = false; };
}

// Pixel Spectral Query & Chart
function queryPixel(e) {
    let img = getActiveImage();
    if (!img) return;
    let rect = img.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    let relX = (e.clientX - rect.left) / rect.width;
    let relY = (e.clientY - rect.top) / rect.height;
    if (relX < 0 || relX > 1 || relY < 0 || relY > 1) return;

    fetch(`/pixel_spectrum?x=${relX.toFixed(4)}&y=${relY.toFixed(4)}`)
        .then(res => res.json())
        .then(data => renderSpectralChart(data))
        .catch(() => {});
}

function renderSpectralChart(data) {
    let panel = document.getElementById('spectral-chart-panel');
    let svg = document.getElementById('spectral-svg');
    let subtitle = document.getElementById('spectral-coords-title');
    if (!panel || !svg || !data.curves) return;
    panel.style.display = 'block';
    if (subtitle) subtitle.innerText = `sampled at relative coordinates X: ${(data.x * 100).toFixed(1)}%, Y: ${(data.y * 100).toFixed(1)}%`;

    let bands = data.bands;
    let wavelengths = [490, 560, 665, 842, 1610];
    let w = 680, h = 210, padLeft = 45, padBottom = 40, padTop = 20, padRight = 20;
    let colors = { input: '#94a3b8', bicubic: '#f59e0b', sen2sr: '#8b5cf6', spectraguard: '#00f2fe' };

    let html = `<svg width="100%" height="100%" viewBox="0 0 ${w} ${h}">`;
    
    // Y-Axis Grid Lines & Values
    for (let l = 0; l <= 4; l++) {
        let val = (l * 0.2).toFixed(1);
        let y = (h - padBottom) - (l * 0.2) * (h - padTop - padBottom);
        html += `<line x1="${padLeft}" y1="${y}" x2="${w-padRight}" y2="${y}" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>`;
        html += `<text x="${padLeft-8}" y="${y+4}" fill="#94a3b8" font-size="10" text-anchor="end" font-family="JetBrains Mono">${val}</text>`;
    }

    // X-Axis Band Labels & Ticks
    bands.forEach((b, i) => {
        let x = padLeft + (i / (bands.length - 1)) * (w - padLeft - padRight);
        html += `<line x1="${x}" y1="${padTop}" x2="${x}" y2="${h-padBottom}" stroke="rgba(255,255,255,0.05)" stroke-dasharray="4"/>`;
        html += `<text x="${x}" y="${h-22}" fill="#f8fafc" font-size="11" font-weight="700" text-anchor="middle" font-family="JetBrains Mono">${b}</text>`;
        html += `<text x="${x}" y="${h-8}" fill="#94a3b8" font-size="9" text-anchor="middle" font-family="JetBrains Mono">${wavelengths[i]}nm</text>`;
    });

    // Draw Curves
    Object.keys(data.curves).forEach(role => {
        let curve = data.curves[role];
        let pts = [];
        curve.forEach((val, i) => {
            if (val !== null && val !== undefined) {
                let x = padLeft + (i / (bands.length - 1)) * (w - padLeft - padRight);
                let y = (h - padBottom) - Math.min(1.0, Math.max(0, val)) * (h - padTop - padBottom);
                pts.push(`${x},${y}`);
            }
        });
        if (pts.length) {
            html += `<polyline points="${pts.join(' ')}" fill="none" stroke="${colors[role]||'#fff'}" stroke-width="2.5"/>`;
            pts.forEach(p => {
                let [px, py] = p.split(',');
                html += `<circle cx="${px}" cy="${py}" r="4" fill="${colors[role]||'#fff'}" stroke="#070a12" stroke-width="1.5"/>`;
            });
        }
    });

    html += `</svg>`;
    svg.innerHTML = html;
}

window.addEventListener('popstate', e => {
    let route = (e.state && e.state.route) ? e.state.route : location.pathname;
    showRouteView(route);
});

document.addEventListener('DOMContentLoaded', () => {
    let savedTheme = localStorage.getItem('sg_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);

    showRouteView(location.pathname);
    updateClock();
    setInterval(updateClock, 1000);

    let drag = false, last = null;
    document.addEventListener('mousemove', e => {
        let img = getActiveImage();
        if (!img) return;
        let rect = img.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        let x = Math.round((e.clientX - rect.left) / rect.width * 100);
        let y = Math.round((e.clientY - rect.top) / rect.height * 100);
        if (x >= 0 && x <= 100 && y >= 0 && y <= 100) {
            let activeView = document.querySelector('.app-view.active');
            if (activeView) {
                let coords = activeView.querySelector('.cursor-coords-badge');
                if (coords) coords.innerText = `Cursor: X:${x}% Y:${y}%`;
            }
        }
    });

    document.addEventListener('wheel', e => {
        let img = getActiveImage();
        if (img && img.contains(e.target)) {
            e.preventDefault();
            zoom(e.deltaY < 0 ? 0.2 : -0.2);
        }
    }, { passive: false });

    document.addEventListener('pointerdown', e => {
        let img = getActiveImage();
        if (img && (img === e.target || img.contains(e.target))) {
            drag = true;
            last = e;
            img.setPointerCapture(e.pointerId);
        }
    });

    document.addEventListener('pointermove', e => {
        if (!drag || !last) return;
        viewState.x += e.clientX - last.clientX;
        viewState.y += e.clientY - last.clientY;
        last = e;
        paint();
    });

    document.addEventListener('pointerup', () => { drag = false; last = null; });
    document.addEventListener('click', e => {
        let img = getActiveImage();
        if (img && (img === e.target || img.contains(e.target))) {
            queryPixel(e);
        }
    });

    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeLightbox();
    });
});
</script>"""


def generate_single_page_app(current_route: str) -> bytes:
    data, error = config()
    error_text = f'<p class="muted">{html.escape(error)}</p>' if error else ""
    info = _scene_info(data)
    status = artifact_state()
    is_ready = status == "Results Ready"
    status_class = "status" if is_ready else "status warn"

    links = "".join(
        f'<a class="nav-link {"active" if href == current_route else ""}" href="{href}" data-route="{href}" onclick="navigateTo(\'{href}\', event)">{icon} <span>{label}</span></a>'
        for href, label, icon, _ in NAV
    )

    header = (
        f'<div class="topbar">'
        f'<div class="topbar-info">'
        f'<span class="topbar-pill">{ICONS["globe"]} <strong>{html.escape(info["scene"])}</strong></span>'
        f'<span class="topbar-pill">{html.escape(info["crs"])}</span>'
        f'<span class="topbar-pill">Resolution: <strong>{info["resolution"]} → 2.5m (4×)</strong></span>'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:14px">'
        f'<span class="topbar-pill" id="utc-clock">00:00:00 UTC</span>'
        f'<button class="theme-btn" onclick="toggleTheme()">{ICONS["sun"]} Theme</button>'
        f'<span class="{status_class}">{status}</span>'
        f'</div>'
        f'</div>'
    )

    # 1. View Home
    view_home = (
        f'<div id="view-home" class="app-view {"active" if current_route == "/" else ""}">'
        f'<div class="eyebrow">Mission overview</div>'
        f'<h1>SpectraGuard-SRM</h1>'
        f'<p class="muted">Sentinel-2 spectral-spatial super-resolution and analytical fusion platform with verifiable internal consistency.</p>'
        f'{error_text}{cards(data)}'
        f'<h2>Generated Pipeline Artifact Inventory</h2>'
        f'<div class="grid-cards">'
    )
    for label, role in (
        ("Original Input", "input"),
        ("Bicubic Baseline", "bicubic"),
        ("SEN2SR Model", "sen2sr"),
        ("SpectraGuard Fused", "spectraguard"),
        ("Validation Metrics", "metrics"),
        ("Uncertainty Map", "uncertainty"),
        ("Run Metadata", "metadata"),
    ):
        avail = artifact(role) is not None
        cls = "good" if avail else "warn"
        text = "Available" if avail else "Not available"
        view_home += (
            f'<div class="metric-card">'
            f'<div class="metric-icon">{ICONS["layer"]}</div>'
            f'<div class="metric-content">'
            f'<small>{label}</small>'
            f'<strong class="{cls}">{text}</strong>'
            f'</div>'
            f'</div>'
        )
    view_home += (
        f'</div>'
        f'<div class="buttons">'
        f'<a href="/scene" onclick="navigateTo(\'/scene\', event)">{ICONS["scene"]} Explore Scene</a>'
        f'<a href="/process" onclick="navigateTo(\'/process\', event)">{ICONS["process"]} Process Workflow</a>'
        f'<a href="/results" onclick="navigateTo(\'/results\', event)">{ICONS["results"]} View Generated Results</a>'
        f'</div>'
        f'</div>'
    )

    # 2. View Scene
    aoi = data.get("data", {}).get("aoi", {})
    formatted_bounds = _format_bounds(info.get("bounds") or aoi)
    view_scene = (
        f'<div id="view-scene" class="app-view {"active" if current_route == "/scene" else ""}">'
        f'<div class="eyebrow">Sentinel-2 measurement granule</div>'
        f'<h1>Explore Scene</h1>'
        f'<p class="fixed">Fixed Processing AOI — defined in project configuration.</p>'
        f'<p class="muted">Natural color visualization (B04 Red, B03 Green, B02 Blue). Interactive zoom, pan, and cursor coordinate tracking.</p>'
        f'{viewer("/scene.png", "Scene preview unavailable", "scene-viewer", "scene-image", _aoi_overlay(data, info))}'
        f'<div class="grid-cards">'
        f'<div class="metric-card"><div class="metric-icon">{ICONS["globe"]}</div><div class="metric-content"><small>Scene ID</small><strong title="{html.escape(info["scene"])}">{html.escape(info["scene"])}</strong></div></div>'
        f'<div class="metric-card"><div class="metric-icon">{ICONS["about"]}</div><div class="metric-content"><small>Acquisition Time</small><strong>{html.escape(info["acquisition"])}</strong></div></div>'
        f'<div class="metric-card"><div class="metric-icon">{ICONS["layer"]}</div><div class="metric-content"><small>Coordinate Reference System</small><strong>{html.escape(info["crs"])}</strong></div></div>'
        f'<div class="metric-card"><div class="metric-icon">{ICONS["chart"]}</div><div class="metric-content"><small>Resolutions (Obs / Work / Target)</small><strong>{html.escape(info["resolution"])} / {get_config(data,"data","working_resolution_m")} / {get_config(data,"data","target_resolution_m")}</strong></div></div>'
        f'<div class="metric-card"><div class="metric-icon">{ICONS["scene"]}</div><div class="metric-content"><small>Fixed AOI Bounds</small><strong>{formatted_bounds}</strong></div></div>'
        f'<div class="metric-card"><div class="metric-icon">{ICONS["shield"]}</div><div class="metric-content"><small>Spectral Bands</small><strong>{html.escape(get_config(data,"data","bands"))}</strong></div></div>'
        f'</div>'
        f'</div>'
    )

    # 3. View Process
    checks = {
        "Input ingestion": artifact("input") is not None,
        "Geospatial preprocessing": artifact("input") is not None,
        "Bicubic baseline": artifact("bicubic") is not None,
        "Spatial SR inference": artifact("sen2sr") is not None,
        "Spectral injection & fusion": artifact("spectraguard") is not None,
        "Consistency projection": artifact("spectraguard") is not None,
        "Validation & uncertainty": bool(_metrics()),
        "FCLS unmixing": artifact("abundance") is not None,
        "Output artifacts ready": status == "Results Ready",
    }
    steps_html = []
    for name, done in checks.items():
        cls = "complete" if done else "missing"
        status_txt = "Available" if done else "Pending"
        steps_html.append(
            f'<div class="step {cls}">'
            f'<div class="step-title">{html.escape(name)}</div>'
            f'<span class="step-status">{status_txt}</span>'
            f'</div>'
        )
    flow = '<div class="arrow">↓</div>'.join(steps_html)
    view_process = (
        f'<div id="view-process" class="app-view {"active" if current_route == "/process" else ""}">'
        f'<div class="eyebrow">Processing overview</div>'
        f'<h1>Process Workflow</h1>'
        f'<p class="muted">Viewing this page does not run ML inference. Pipeline status is derived strictly from verified project disk artifacts.</p>'
        f'{cards(data)}<div class="workflow">{flow}</div>'
        f'</div>'
    )

    # 4. View Results
    products = (
        ("input", "Original Sentinel-2 (10m)"),
        ("bicubic", "Bicubic Baseline (2.5m)"),
        ("sen2sr", "Standalone SEN2SR (2.5m)"),
        ("spectraguard", "SpectraGuard Fused (2.5m)"),
    )
    tabs_prod = "".join(
        f'<button class="tab tab-product {"active" if i == 3 else ""}" onclick="selectResult(\'{role}\',this)">{title}</button>'
        for i, (role, title) in enumerate(products)
    )
    tabs_mode = (
        f'<button class="tab tab-mode active" onclick="selectMode(\'rgb\',this)">🌿 True Color (RGB)</button>'
        f'<button class="tab tab-mode" onclick="selectMode(\'nir\',this)">🔴 False Color (NIR)</button>'
        f'<button class="tab tab-mode" onclick="selectMode(\'swir\',this)">💧 SWIR Composite</button>'
    )

    metrics = _metrics()
    m_bicubic = model_metrics("bicubic")
    m_sen2sr = model_metrics("sen2sr")
    m_spectraguard = model_metrics("spectraguard")

    def _fmt_sam(m):
        v = m.get("sam_mean")
        return f"{v:.6f} rad" if isinstance(v, (int, float)) else "N/A"

    def _fmt_chips(m, key):
        d = m.get(key)
        if isinstance(d, dict) and d:
            return " ".join(f'<span class="chip">{b}: {v:.4f}</span>' for b, v in d.items())
        return "N/A"

    sam_bicubic = _fmt_sam(m_bicubic)
    sam_sen2sr = _fmt_sam(m_sen2sr)
    sam_sg = _fmt_sam(m_spectraguard)

    rmse_bicubic = _fmt_chips(m_bicubic, "per_band_rmse")
    rmse_sen2sr = _fmt_chips(m_sen2sr, "per_band_rmse")
    rmse_sg = _fmt_chips(m_spectraguard, "per_band_rmse")

    mae_bicubic = _fmt_chips(m_bicubic, "per_band_mae")
    mae_sen2sr = _fmt_chips(m_sen2sr, "per_band_mae")
    mae_sg = _fmt_chips(m_spectraguard, "per_band_mae")

    rows = (
        f'<tr><th>Spectral Angle Mapper (SAM ↓)</th><td>{sam_bicubic}</td><td>{sam_sen2sr}</td><td><strong class="good">{sam_sg}</strong></td></tr>'
        f'<tr><th>Per-Band RMSE ↓</th><td>{rmse_bicubic}</td><td>{rmse_sen2sr}</td><td>{rmse_sg}</td></tr>'
        f'<tr><th>Per-Band MAE ↓</th><td>{mae_bicubic}</td><td>{mae_sen2sr}</td><td>{mae_sg}</td></tr>'
        f'<tr><th>PSNR ↑</th><td>N/A</td><td>N/A</td><td><span class="muted" style="font-size:12px">N/A (Internal Consistency Mode)</span></td></tr>'
        f'<tr><th>SSIM ↑</th><td>N/A</td><td>N/A</td><td><span class="muted" style="font-size:12px">N/A (Internal Consistency Mode)</span></td></tr>'
    )

    generated_plots_html = ""
    for plot_key, plot_title in GENERATED_PLOTS.items():
        plot_file = OUTPUTS / f"{plot_key}.png"
        if plot_file.is_file():
            generated_plots_html += (
                f'<div class="plot-card">'
                f'<h3><span>{plot_title}</span>'
                f'<button class="button" style="padding:5px 12px;font-size:12px" onclick="openLightbox(\'/output_plot/{plot_key}.png\')">{ICONS["expand"]} Full View</button>'
                f'</h3>'
                f'<img src="/output_plot/{plot_key}.png" alt="{plot_title}" onclick="openLightbox(\'/output_plot/{plot_key}.png\')">'
                f'</div>'
            )

    swipe_curtain = (
        f'<div class="swipe-container" id="swipe-box">'
        f'<span class="swipe-badge-left">Original Sentinel-2 (10m)</span>'
        f'<span class="swipe-badge-right">SpectraGuard Fused (2.5m)</span>'
        f'<img src="/result/spectraguard.png" class="swipe-after" alt="SpectraGuard Fused">'
        f'<div class="swipe-before-wrapper" id="swipe-before-wrap">'
        f'<img src="/result/input.png" class="swipe-before" alt="Original Input">'
        f'</div>'
        f'<div class="swipe-handle" id="swipe-handle">'
        f'<div class="swipe-circle">◄ ►</div>'
        f'</div>'
        f'</div>'
    )

    spectral_chart_panel = (
        f'<div id="spectral-chart-panel" class="spectral-panel" style="display:none">'
        f'<h3 style="font-size:16px;font-weight:700;margin-bottom:4px">Pixel Reflectance Spectrum Inspector</h3>'
        f'<p class="muted" id="spectral-coords-title" style="font-size:12.5px;margin-bottom:12px">Click any pixel location on the interactive imagery canvas to sample its multi-band reflectance signature.</p>'
        f'<div style="display:flex;gap:16px;margin-bottom:12px;font-size:12px;font-family:var(--code-font);flex-wrap:wrap">'
        f'<span style="color:#94a3b8">● Original (10m)</span>'
        f'<span style="color:#f59e0b">● Bicubic Baseline</span>'
        f'<span style="color:#8b5cf6">● Standalone SEN2SR</span>'
        f'<span style="color:#00f2fe;font-weight:bold">● SpectraGuard Fused</span>'
        f'</div>'
        f'<div class="spectral-chart-wrapper" id="spectral-svg"></div>'
        f'</div>'
    )

    view_results = (
        f'<div id="view-results" class="app-view {"active" if current_route == "/results" else ""}">'
        f'<div class="eyebrow">Artifact comparison</div>'
        f'<h1>Generated Pipeline Results</h1>'
        f'<p class="muted">Side-by-side comparative views, interactive swipe curtain slider, and multi-spectral band composite selection.</p>'
        f'<h2>Interactive 10m vs 2.5m Swipe Comparison Curtain</h2>'
        f'<p class="muted">Drag the vertical divider handle left and right across the scene to compare 10m Sentinel-2 against 2.5m SpectraGuard Super-Resolution:</p>'
        f'{swipe_curtain}'
        f'<h2>Multi-Band Composite & Focus Viewer</h2>'
        f'<div class="buttons">{tabs_prod}</div>'
        f'<div class="buttons">{tabs_mode}</div>'
        f'<p id="sen2sr-swir-note" class="muted" style="display:none;font-size:12px;color:var(--accent-gold);margin-bottom:12px">ℹ️ SEN2SR model operates on 4 VNIR bands (B02, B03, B04, B08). SWIR band B11 is preserved directly from original Sentinel-2 observation.</p>'
        f'{viewer("/result/spectraguard_rgb.png", "Result preview not available.", "scientific-viewer", "scientific-image")}'
        f'{spectral_chart_panel}'
        f'<h2>Generated Pipeline Comparison Plots</h2>'
        f'{generated_plots_html}'
        f'<h2>Quantitative Evaluation Metrics</h2>'
        f'<p class="muted">Validation evaluation against degraded input observation. Reference mode: <strong>{html.escape(str(metrics.get("reference_type","internal_consistency")))}</strong></p>'
        f'<table class="table"><tr><th>Metric</th><th>Bicubic</th><th>Standalone SEN2SR</th><th>SpectraGuard Fused</th></tr>{rows}</table>'
        f'</div>'
    )

    # 5. View Analysis
    comparison = "".join(
        f'<div class="card">'
        f'<small>{title}</small>'
        f'<img src="/result/{role}.png" alt="{title}" onclick="openLightbox(\'/result/{role}.png\')">'
        f'<strong style="font-size:14px;margin-top:6px">{"Available" if artifact(role) else "Not available"}</strong>'
        f'</div>'
        for role, title in (
            ("input", "Original"),
            ("bicubic", "Bicubic"),
            ("sen2sr", "SEN2SR"),
            ("spectraguard", "SpectraGuard"),
        )
    )
    maps = '<div class="grid-cards" style="grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));">'
    for role, title, desc in (
        ("sam_map", "Spectral Angle Mapper (SAM) Map", "Pixel-wise spectral deviation in radians."),
        ("uncertainty", "Composite Uncertainty Map", "Combined variance and spectral reconstruction uncertainty."),
    ):
        avail = artifact(role) is not None
        maps += (
            f'<div class="metric-card" style="flex-direction:column">'
            f'<small>{title}</small>'
            f'<img class="map-image" src="/map/{role}.png" alt="{title}" onclick="openLightbox(\'/map/{role}.png\')">'
            f'<strong class="{"good" if avail else "warn"}" style="margin-top:10px;font-size:14px">{"Available" if avail else "Not available"}</strong>'
            f'<p class="muted" style="font-size:12px;margin-top:4px">{desc}</p>'
            f'</div>'
        )
    maps += "</div>"

    ground_plots = ""
    for plot_key in ("ground_detail_comparison", "ground_detail_true_color", "ground_detail_comparison_nir"):
        if (OUTPUTS / f"{plot_key}.png").is_file():
            title = GENERATED_PLOTS.get(plot_key, plot_key)
            ground_plots += (
                f'<div class="plot-card">'
                f'<h3><span>{title}</span>'
                f'<button class="button" style="padding:5px 12px;font-size:12px" onclick="openLightbox(\'/output_plot/{plot_key}.png\')">{ICONS["expand"]} Full View</button>'
                f'</h3>'
                f'<img src="/output_plot/{plot_key}.png" alt="{title}" onclick="openLightbox(\'/output_plot/{plot_key}.png\')">'
                f'</div>'
            )

    view_analysis = (
        f'<div id="view-analysis" class="app-view {"active" if current_route == "/analysis" else ""}">'
        f'<div class="eyebrow">Artifact-backed analysis</div>'
        f'<h1>Scientific Analysis</h1>'
        f'<p class="muted">Quantitative spatial-spectral error maps, uncertainty estimates, and high-resolution ground detail plots.</p>'
        f'<div class="comparison">{comparison}</div>'
        f'<h2>Spatial & Spectral Uncertainty Maps</h2>'
        f'{maps}'
        f'<h2>Generated Ground Detail Comparison Plots</h2>'
        f'{ground_plots}'
        f'</div>'
    )

    # 6. View Validation
    LABEL_MAP = {
        "sam_mean": "Spectral Angle Mapper (SAM Mean)",
        "per_band_rmse": "Per-Band Root Mean Square Error (RMSE)",
        "per_band_mae": "Per-Band Mean Absolute Error (MAE)",
        "reference_type": "Validation Reference Mode",
        "psnr": "Peak Signal-to-Noise Ratio (PSNR)",
        "ssim": "Structural Similarity Index (SSIM)",
        "ergas": "Relative Dimensionless Global Error (ERGAS)",
        "band_names": "Evaluated Spectral Bands",
    }
    val_rows = []
    for raw_key, value in metrics.items():
        key_label = LABEL_MAP.get(raw_key.lower(), raw_key.replace("_", " ").title())
        if value is None:
            val_str = '<span class="muted" style="font-size:12px">N/A (Requires High-Resolution Ground Truth Reference)</span>'
        elif isinstance(value, dict):
            val_str = " ".join(f'<span class="chip">{b}: {v:.4f}</span>' for b, v in value.items())
        elif isinstance(value, float):
            val_str = f'<strong class="good">{value:.6f} rad</strong>' if "sam" in raw_key.lower() else f"{value:.6f}"
        elif isinstance(value, list):
            val_str = " ".join(f'<span class="chip">{b}</span>' for b in value)
        else:
            val_str = f"<strong>{html.escape(str(value))}</strong>"
        val_rows.append(f"<tr><th>{html.escape(key_label)}</th><td>{val_str}</td></tr>")

    val_values_html = (
        "".join(val_rows) if val_rows else '<tr><td colspan="2">Validation metrics not available yet.</td></tr>'
    )
    view_validation = (
        f'<div id="view-validation" class="app-view {"active" if current_route == "/validation" else ""}">'
        f'<div class="eyebrow">Scientific validation</div>'
        f'<h1>Validation Report</h1>'
        f'<p class="muted">Internal-consistency validation verifies that downsampling the super-resolved product reproduces original Sentinel-2 observations.</p>'
        f'<div class="plot-card" style="margin-top:20px;margin-bottom:24px">'
        f'<h3 style="margin-bottom:8px">Validation Methodology: Internal Consistency Verification</h3>'
        f'<p class="muted" style="font-size:13.5px"><strong>2.5m Super-Resolved Output → Downsample to 10m → Calculate Spectral Angle Mapper & Band Errors against Original Sentinel-2 Input Observation</strong></p>'
        f'</div>'
        f'<table class="table">{val_values_html}</table>'
        f'<p class="muted" style="margin-top:16px">Note: Internal consistency validation operates without requiring an independent high-resolution ground truth image.</p>'
        f'</div>'
    )

    # 7. View FCLS
    path = artifact("abundance")
    names = "Not available"
    matrix = _endmember_matrix()
    if path:
        try:
            import numpy as np

            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as values:
                    names = (
                        ", ".join(map(str, values["endmember_names"].tolist()))
                        if "endmember_names" in values.files
                        else "Names stored in manifest"
                    )
                    dimensions = str(values["S"].shape) if "S" in values.files else "Shape not available"
            else:
                dimensions = str(np.load(path, allow_pickle=False).shape)
        except (OSError, ValueError, KeyError):
            names = "Artifact could not be read"

    if matrix:
        bands, matrix_names, values = matrix
        matrix_rows = "".join(
            f"<tr><th>{html.escape(band)}</th>"
            + "".join(f'<td><span class="chip">{val:.4f}</span></td>' for val in row)
            + "</tr>"
            for band, row in zip(bands, values)
        )
        matrix_view = (
            f"<h2>Endmember Spectral Response Matrix (A)</h2>"
            f'<p class="muted">Sentinel-2C band response weights for USGS spectral endmembers used during FCLS unmixing.</p>'
            f'<table class="table"><tr><th>Band</th>'
            + "".join(f"<th>{html.escape(name)}</th>" for name in matrix_names)
            + f"</tr>{matrix_rows}</table>"
        )
    else:
        matrix_view = '<p class="muted">Endmember matrix source unavailable.</p>'

    view_fcls = (
        f'<div id="view-fcls" class="app-view {"active" if current_route == "/fcls" else ""}">'
        f'<div class="eyebrow">Material interpretation</div>'
        f'<h1>FCLS / Endmember Unmixing</h1>'
        f'<p class="muted">Fully Constrained Least Squares (FCLS) estimates sub-pixel endmember material abundances under non-negativity and sum-to-one constraints.</p>'
        f'<div class="grid-cards">'
        f'<div class="metric-card"><div class="metric-icon">{ICONS["fcls"]}</div><div class="metric-content"><small>Endmember Materials</small><strong>{html.escape(names)}</strong></div></div>'
        f'<div class="metric-card"><div class="metric-icon">{ICONS["layer"]}</div><div class="metric-content"><small>Abundance Cube Dimensions</small><strong>{html.escape(locals().get("dimensions", "Configured"))}</strong></div></div>'
        f'</div>'
        f'{matrix_view}'
        f'</div>'
    )

    # 8. View About
    flow_steps = (
        "Sentinel-2 L2A Input",
        "Ingestion & AOI",
        "Bicubic Baseline",
        "SEN2SR Detail Prior",
        "Covariance-based Spectral Injection",
        "Consistency Projection",
        "Validation & Uncertainty",
        "FCLS Unmixing",
    )
    flow_str = " → ".join(flow_steps)
    view_about = (
        f'<div id="view-about" class="app-view {"active" if current_route == "/about" else ""}">'
        f'<div class="eyebrow">Project overview</div>'
        f'<h1>About SpectraGuard-SRM</h1>'
        f'<p class="muted">SpectraGuard-SRM is an advanced Sentinel-2 super-resolution and analytical spectral-spatial fusion system.</p>'
        f'<div class="plot-card" style="margin-top:24px">'
        f'<h3 style="margin-bottom:8px">What is SpectraGuard-SRM?</h3>'
        f'<p class="muted" style="font-size:14px;line-height:1.7;margin-bottom:16px">SpectraGuard-SRM uses a swappable super-resolution model and adds spectral-spatial analysis, measurement/degradation consistency, uncertainty/anomaly analysis, and endmember-based interpretation around the SR output.</p>'
        f'<h3 style="margin-bottom:8px;font-size:14px">System Data Pipeline Architecture</h3>'
        f'<strong style="font-size:13.5px;color:var(--accent-cyan-bright);line-height:1.8;display:block">{html.escape(flow_str)}</strong>'
        f'</div>'
        f'</div>'
    )

    lightbox = (
        f'<div id="lightbox-modal" class="lightbox" onclick="closeLightbox()">'
        f'<button class="lightbox-close" onclick="closeLightbox()">✕</button>'
        f'<img id="lightbox-img" src="" alt="Full resolution view" onclick="event.stopPropagation()">'
        f'</div>'
    )

    page_title = next((label for href, label, _, _ in NAV if href == current_route), "Dashboard")

    return (
        f"<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{html.escape(page_title)} · SpectraGuard-SRM Science Center</title><style>{CSS}</style></head><body><div class=\"layout\">"
        f'<aside class="side">'
        f'<div class="brand">Spectra<b>Guard</b> <span class="brand-badge">SRM</span></div>'
        f'<div class="sub">Scientific Super-Resolution</div>'
        f'<nav class="nav">{links}</nav>'
        f'<div class="side-footer"><div class="pulse-dot"></div><span>Server Status: <strong>Online</strong> (Port 8080)</span></div>'
        f'</aside>'
        f'<div class="main-wrapper">{header}<main class="main">{view_home}{view_scene}{view_process}{view_results}{view_analysis}{view_validation}{view_fcls}{view_about}</main></div>'
        f'</div>{lightbox}{SCRIPT}</body></html>'
    ).encode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        route = unquote(parsed.path)
        params = parse_qs(parsed.query)

        try:
            if route == "/scene.png":
                body, kind = raster_preview(config()[0]), "image/png"
            elif route == "/pixel_spectrum":
                x = float(params.get("x", [0.5])[0])
                y = float(params.get("y", [0.5])[0])
                body = json.dumps(sample_pixel_spectrum(x, y)).encode("utf-8")
                kind = "application/json"
            elif route.startswith("/result/") and route.endswith(".png"):
                name = route[8:-4]
                if "_" in name:
                    role, mode = name.rsplit("_", 1)
                    if mode not in ("rgb", "nir", "swir"):
                        role, mode = name, "rgb"
                else:
                    role, mode = name, "rgb"

                if role not in ("input", "bicubic", "sen2sr", "spectraguard") or artifact(role) is None:
                    raise FileNotFoundError()
                body, kind = _rgb_preview(role, mode), "image/png"
            elif route.startswith("/map/") and route.endswith(".png"):
                role = route[5:-4]
                if role not in ("sam_map", "uncertainty") or artifact(role) is None:
                    raise FileNotFoundError()
                body, kind = _map_preview(role), "image/png"
            elif route.startswith("/output_plot/") and route.endswith(".png"):
                plot_name = route[13:-4]
                body, kind = output_plot_bytes(plot_name), "image/png"
            elif route in ("/", "/scene", "/process", "/results", "/analysis", "/validation", "/fcls", "/about"):
                body, kind = generate_single_page_app(route), "text/html; charset=utf-8"
            else:
                raise KeyError(route)

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass
        except KeyError:
            self.send_error(HTTPStatus.NOT_FOUND, "Page not found")
        except (FileNotFoundError, ValueError, OSError):
            self.send_error(HTTPStatus.NOT_FOUND, "Resource unavailable")

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    print(f"Open http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
