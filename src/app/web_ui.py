"""Read-only scientific dashboard for existing SpectraGuard-SRM artifacts."""
from __future__ import annotations

import argparse
import html
import json
import re
import struct
import zlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "default.yaml"
OUTPUTS = ROOT / "data" / "outputs"
NAV = (
    ("/", "Home"), ("/scene", "Explore Scene"), ("/process", "Process"),
    ("/results", "Results"), ("/analysis", "Analysis"),
    ("/validation", "Validation"), ("/fcls", "FCLS / Endmembers"),
    ("/about", "About"),
)
ROLE_CANDIDATES = {
    "input": (OUTPUTS / "validation" / "original.npz", OUTPUTS / "validation" / "original.npy"),
    "bicubic": (OUTPUTS / "bicubic" / "bicubic.npz", OUTPUTS / "bicubic" / "bicubic.npy"),
    "sen2sr": (OUTPUTS / "sr" / "spatial_prediction.npz",),
    "spectraguard": (OUTPUTS / "fused" / "fused.npz", OUTPUTS / "fused" / "fused.npy"),
    "metrics": (OUTPUTS / "validation" / "metrics.json", OUTPUTS / "validation" / "metrics.npz"),
    "sam_map": (OUTPUTS / "validation" / "sam_map.npy",),
    "uncertainty": (OUTPUTS / "uncertainty" / "uncertainty.npz", OUTPUTS / "uncertainty" / "uncertainty.npy"),
    "abundance": (OUTPUTS / "abundance" / "abundance_sample.npz", OUTPUTS / "abundance" / "abundance.npy"),
    "metadata": (OUTPUTS / "reports" / "run_metadata.json",),
}


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
    """Safely inspect an artifact without assuming the first stored array is useful."""
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
                preferred = ("data", "cube", "fused", "prediction", "spatial_prediction", "bicubic", "original", "U", "S")
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
    if path.suffix == ".npz":
        try:
            import numpy as np
            with np.load(path, allow_pickle=False) as values:
                return {key: values[key].tolist() for key in values.files}
        except (OSError, ValueError):
            return {}
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        return values if isinstance(values, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


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


def _stretch(array):
    import numpy as np
    valid = array[np.isfinite(array) & (array > 0)]
    if not valid.size:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(valid, (2, 98))
    return np.clip((array - low) * 255 / max(high - low, 1e-6), 0, 255).astype(np.uint8)


def as_png(rgb) -> bytes:
    height, width, _ = rgb.shape
    raw = b"".join(b"\0" + rgb[y].tobytes() for y in range(height))

    def chunk(name, body):
        return struct.pack(">I", len(body)) + name + body + struct.pack(
            ">I", zlib.crc32(name + body) & 0xffffffff
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _rgb_preview(role: str) -> bytes:
    import numpy as np
    data, bands = _npz_data(role)
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 2:
        channels = [array] * 3
    elif array.ndim == 3:
        if bands and all(name in bands for name in ("B04", "B03", "B02")):
            channels = [array[bands.index(name)] for name in ("B04", "B03", "B02")]
        elif array.shape[0] <= 32:
            channels = list(array[:3])
        else:
            channels = list(array[..., :3].transpose(2, 0, 1))
        channels = (channels + [channels[-1]] * 3)[:3]
    else:
        raise ValueError("Artifact image array has unsupported dimensions")
    return as_png(np.dstack([_stretch(channel) for channel in channels]))


def _map_preview(role: str) -> bytes:
    import numpy as np
    path = artifact(role)
    if path is None:
        raise FileNotFoundError(f"{role} artifact is not available")
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as values:
            key = "U" if role == "uncertainty" and "U" in values.files else "sam"
            if key not in values.files:
                raise ValueError(f"{role} artifact has no displayable map")
            array = values[key]
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{role} map has shape {array.shape}; expected a 2-D map")
    image = _stretch(array)
    return as_png(np.dstack([image, image, image]))


def _scene_info(data: dict) -> dict:
    scene_value = get_config(data, "data", "scene_path", default="")
    scene = Path(scene_value)
    if not scene.is_absolute():
        scene = ROOT / scene
    info = {"scene": scene.name or "Not available", "crs": "Not available",
            "resolution": get_config(data, "data", "working_resolution_m"),
            "acquisition": "Not available", "bounds": None}
    try:
        import rasterio
        band = next(scene.rglob("*_B02_10m.jp2"))
        with rasterio.open(band) as source:
            info["crs"] = source.crs.to_string() if source.crs else "Not available"
            info["resolution"] = f"{abs(source.transform.a):g} m"
            info["bounds"] = [source.bounds.left, source.bounds.bottom, source.bounds.right, source.bounds.top]
    except (OSError, StopIteration, ImportError):
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
    import numpy as np
    import rasterio
    paths = scene_bands(data)
    if paths is None:
        raise FileNotFoundError("Input scene unavailable")
    arrays = []
    for path in paths:
        with rasterio.open(path) as source:
            arrays.append(source.read(1, out_shape=(1024, 1024), masked=True).filled(0).astype(np.float32) / 10000.0)
    return as_png(np.dstack([_stretch(array) for array in arrays]))


def cards(data: dict) -> str:
    fields = (
        ("Input resolution", get_config(data, "data", "working_resolution_m")),
        ("Target resolution", get_config(data, "data", "target_resolution_m")),
        ("Input bands", get_config(data, "data", "bands")),
        ("SR model", get_config(data, "sr", "model")),
        ("Scale", get_config(data, "sr", "scale")),
        ("Device", get_config(data, "sr", "device")),
        ("Fixed AOI", str(data.get("data", {}).get("aoi", "Not configured"))),
        ("Sampling steps", get_config(data, "sr", "sampling_steps")),
    )
    return '<div class="grid">' + "".join(
        f'<div class="card"><small>{html.escape(name)}</small><strong>{html.escape(value)}</strong></div>'
        for name, value in fields
    ) + "</div>"


CSS = """*{box-sizing:border-box}body{margin:0;background:#07111a;color:#dbe8ed;font:14px Inter,Segoe UI,sans-serif}.layout{display:flex;min-height:100vh}.side{width:245px;padding:28px 15px;background:#0b1823;border-right:1px solid #1b3542}.brand{font-size:21px;font-weight:750;color:#fff}.brand b{color:#52d4c1}.sub{font-size:11px;color:#86a7b5;margin:7px 0 27px}.nav a{display:block;padding:10px 13px;color:#b6cbd4;text-decoration:none;border-radius:7px}.nav a:hover,.nav .active{background:#12323d;color:#69e4d2}.main{width:min(1500px,100%);padding:36px 48px}.eyebrow{font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:#55cebd}h1{font-size:38px;margin:9px 0 12px}.muted{color:#99aeb9;line-height:1.6}.status{display:inline-block;border-radius:16px;padding:6px 10px;color:#73e0d1;background:#123c40;font-size:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:13px;margin:25px 0}.card{background:#0d1d29;border:1px solid #1a3542;border-radius:10px;padding:16px}.card small{display:block;color:#8ca4ae;text-transform:uppercase;font-size:10px;letter-spacing:1px}.card strong{display:block;margin-top:8px;word-break:break-word}.buttons a,.tab,.button{display:inline-block;border:0;border-radius:6px;background:#197a75;color:#fff;text-decoration:none;padding:10px 14px;margin:10px 8px 0 0;cursor:pointer}.tab{background:#132c37;color:#c8dbe0}.tab.active{background:#197a75}.viewer{min-height:430px;background:#02090d;border:1px solid #1b3642;border-radius:10px;overflow:hidden;position:relative;touch-action:none}.viewer img{width:100%;height:430px;object-fit:contain;display:block;transform-origin:center;cursor:grab}.viewer .empty{padding:45px;color:#a7bac2}.controls{margin:12px 0}.controls button{border:1px solid #294956;background:#132b36;color:#dcebef;border-radius:5px;padding:7px 10px;margin-right:6px;cursor:pointer}.fixed{color:#f0ca7a}.workflow{max-width:800px;margin:25px 0}.step{padding:13px 16px;margin:7px 0;border-left:3px solid #56cbbc;background:#0d1d29;border-radius:4px}.complete{border-left-color:#73e0d1}.missing{border-left-color:#8b6d58;color:#aebbc0}.arrow{margin-left:20px;color:#56cbbc}.aoi-scene{position:relative}.aoi-scene img{width:100%;height:430px;object-fit:contain;display:block}.aoi-box{position:absolute;border:2px solid #f0ca7a;pointer-events:none}.aoi-box span{position:absolute;top:5px;left:5px;white-space:nowrap;background:#07111add;padding:4px 7px;color:#f0ca7a;font-size:11px}.comparison{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.comparison .card{padding:10px}.comparison img{width:100%;height:210px;object-fit:contain;background:#02090d}.table{width:100%;border-collapse:collapse}.table th,.table td{text-align:left;padding:10px;border-bottom:1px solid #1b3542}.table th{color:#86a7b5;font-size:11px}.small{font-size:12px}.good{color:#73e0d1}.warn{color:#f0ca7a}@media(max-width:900px){.comparison{grid-template-columns:repeat(2,1fr)}}@media(max-width:720px){.side{width:65px;padding:20px 8px}.brand,.sub,.nav a{font-size:0}.nav a{height:37px}.main{padding:28px 19px}h1{font-size:31px}.comparison{grid-template-columns:1fr}}
"""

SCRIPT = """<script>
let view={scale:1,x:0,y:0};function image(){return document.querySelector('#scientific-image')}
function paint(){let i=image();if(i)i.style.transform=`translate(${view.x}px,${view.y}px) scale(${view.scale})`}
function zoom(delta){view.scale=Math.min(8,Math.max(.35,view.scale+delta));paint()}
function resetView(){view={scale:1,x:0,y:0};paint()}
function selectResult(role,button){document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));button.classList.add('active');let i=image();i.src='/result/'+role+'.png';resetView()}
document.addEventListener('DOMContentLoaded',()=>{let i=image();if(!i)return;let drag=false,last;i.addEventListener('wheel',e=>{e.preventDefault();zoom(e.deltaY<0?.15:-.15)},{passive:false});i.addEventListener('pointerdown',e=>{drag=true;last=e;i.setPointerCapture(e.pointerId)});i.addEventListener('pointermove',e=>{if(!drag)return;view.x+=e.clientX-last.clientX;view.y+=e.clientY-last.clientY;last=e;paint()});i.addEventListener('pointerup',()=>drag=false)})
</script>"""


def shell(title: str, body: str, route: str) -> bytes:
    links = "".join(
        f'<a class="{"active" if href == route else ""}" href="{href}">{label}</a>'
        for href, label in NAV
    )
    return (
        f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)} · SpectraGuard-SRM</title><style>{CSS}</style></head><body><div class=\"layout\">"
        f'<aside class="side"><div class="brand">Spectra<b>Guard</b>-SRM</div><div class="sub">Scientific super-resolution</div>'
        f'<nav class="nav">{links}</nav></aside><main class="main">{body}</main></div>{SCRIPT}</body></html>'
    ).encode()


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
    return f'<div class="aoi-box" style="left:{x:.3f}%;top:{y:.3f}%;width:{width:.3f}%;height:{height:.3f}%"><span>Fixed Processing AOI</span></div>'


def page(route: str) -> bytes:
    data, error = config()
    error_text = f'<p class="muted">{html.escape(error)}</p>' if error else ""
    info = _scene_info(data)
    status = artifact_state()
    if route == "/":
        body = f'<div class="eyebrow">Mission overview</div><h1>SpectraGuard-SRM</h1><p class="muted">Sentinel-2 spectral-spatial super-resolution and fusion with traceable scientific outputs.</p><span class="status">{status}</span>{error_text}{cards(data)}<div class="grid">'
        for label, role in (("Original Input", "input"), ("Bicubic", "bicubic"), ("SEN2SR", "sen2sr"), ("SpectraGuard", "spectraguard"), ("Validation", "metrics"), ("Uncertainty", "uncertainty"), ("FCLS", "abundance")):
            body += f'<div class="card"><small>{label}</small><strong class="{"good" if artifact(role) else "warn"}">{"Available" if artifact(role) else "Not available"}</strong></div>'
        body += '</div><div class="buttons"><a href="/scene">Explore Scene</a><a href="/process">Process</a><a href="/results">Results</a></div>'
        return shell("Home", body, route)
    if route == "/scene":
        aoi = data.get("data", {}).get("aoi", {})
        body = f'<div class="eyebrow">Sentinel-2 measurement</div><h1>Explore Scene</h1><p class="fixed">Fixed Processing AOI — configuration-defined and not editable.</p><p class="muted">Natural color: B04 → Red, B03 → Green, B02 → Blue. Pan, zoom, and reset are presentation-only.</p><div class="aoi-scene"><img id="scientific-image" src="/scene.png" alt="Sentinel-2 natural color">{_aoi_overlay(data, info)}</div><div class="controls"><button onclick="zoom(-.15)">−</button><button onclick="zoom(.15)">+</button><button onclick="resetView()">Fit / reset</button></div><div class="grid"><div class="card"><small>Scene</small><strong>{html.escape(info["scene"])}</strong></div><div class="card"><small>Acquisition</small><strong>{html.escape(info["acquisition"])}</strong></div><div class="card"><small>CRS</small><strong>{html.escape(info["crs"])}</strong></div><div class="card"><small>Original / Working / Target</small><strong>{html.escape(info["resolution"])} / {get_config(data,"data","working_resolution_m")} / {get_config(data,"data","target_resolution_m")}</strong></div><div class="card"><small>Fixed AOI coordinates</small><strong>{html.escape(str(aoi))}</strong></div><div class="card"><small>Bands</small><strong>{html.escape(get_config(data,"data","bands"))}</strong></div></div>'
        return shell("Explore Scene", body, route)
    if route == "/process":
        checks = {
            "Input ingestion": artifact("input"), "Preprocessing": artifact("input"),
            "Bicubic baseline": artifact("bicubic"), "Spatial representation": artifact("sen2sr"),
            "SEN2SR inference": artifact("sen2sr"), "Spectral fusion": artifact("spectraguard"),
            "Safety / consistency": artifact("spectraguard"), "Validation": bool(_metrics()),
            "FCLS / endmembers": artifact("abundance") is not None,
            "Output artifacts": status == "Results Ready",
        }
        flow = "".join(f'<div class="step {"complete" if done else "missing"}">{name} — {"available" if done else "not available"}</div><div class="arrow">↓</div>' for name, done in checks.items()).rstrip('<div class="arrow">↓</div>')
        return shell("Process", f'<div class="eyebrow">Processing overview</div><h1>Process</h1><span class="status">{status}</span><p class="muted">Opening this page never runs inference. Status is derived only from existing artifacts.</p>{cards(data)}<div class="workflow">{flow}</div>', route)
    if route == "/results":
        products = (("input", "Original Sentinel-2"), ("bicubic", "Bicubic"), ("sen2sr", "Standalone SEN2SR"), ("spectraguard", "SpectraGuard"))
        side = "".join(f'<div class="card"><small>{title}</small><img src="/result/{role}.png" alt="{title}"><strong>{("Available" if artifact(role) else "Result not available")}</strong></div>' for role, title in products)
        tabs = "".join(f'<button class="tab {"active" if i == 0 else ""}" onclick="selectResult(\'{role}\',this)">{title}</button>' for i, (role, title) in enumerate(products))
        metrics = _metrics()
        rows = "".join(f'<tr><th>{label}</th><td>N/A</td><td>N/A</td><td>{html.escape(str(metrics.get(key, "N/A"))) if key in metrics else "N/A"}</td></tr>' for label, key in (("RMSE", "per_band_rmse"), ("SAM ↓", "sam_mean"), ("PSNR ↑", "psnr"), ("SSIM ↑", "ssim"), ("ERGAS ↓", "ergas")))
        body = f'<div class="eyebrow">Artifact comparison</div><h1>Results</h1><span class="status">{status}</span><p class="muted">Side-by-side comparison uses actual saved artifacts. Focus View remains available below.</p><div class="comparison">{side}</div><h2>Focus View</h2><div class="buttons">{tabs}</div>{viewer("/result/input.png","Result not available.")}<h2>Quantitative Comparison</h2><p class="muted">Only values present in the saved validation artifact are shown. Reference type: {html.escape(str(metrics.get("reference_type","N/A")))}</p><table class="table"><tr><th>Metric</th><th>Bicubic</th><th>SEN2SR</th><th>SpectraGuard</th></tr>{rows}</table>'
        return shell("Results", body, route)
    if route == "/analysis":
        comparison = "".join(f'<div class="card"><small>{title}</small><img src="/result/{role}.png" alt="{title}"><strong>{"Available" if artifact(role) else "Not available"}</strong></div>' for role, title in (("input","Original"),("bicubic","Bicubic"),("sen2sr","SEN2SR"),("spectraguard","SpectraGuard")))
        maps = '<div class="grid">'
        for role, title in (("sam_map", "SAM map"), ("uncertainty", "Uncertainty")):
            maps += f'<div class="card"><small>{title}</small><img class="map-image" src="/map/{role}.png" alt="{title}"><p class="muted">{"Available" if artifact(role) else "Analysis artifact not available"}</p></div>'
        maps += "</div>"
        return shell("Analysis", f'<div class="eyebrow">Artifact-backed analysis</div><h1>Analysis</h1><p class="muted">Only saved scientific artifacts are visualized; no values are calculated here.</p><div class="comparison">{comparison}</div><h2>Spectral comparison</h2><div class="card"><p class="muted">Saved cubes contain these project bands where metadata is present: B02, B03, B04, B08, B11. Pixel-sample spectral curves are not stored as a dedicated artifact.</p></div><h2>Maps</h2>{maps}', route)
    if route == "/validation":
        metrics = _metrics()
        values = "".join(f'<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>' for key, value in metrics.items()) or '<tr><td colspan="2">Validation metrics not available yet.</td></tr>'
        return shell("Validation", f'<div class="eyebrow">Scientific validation</div><h1>Validation</h1><p class="muted">Operational internal consistency is distinct from external HR-reference evaluation.</p><table class="table">{values}</table><p class="warn">External HR validation not available for this run.</p>', route)
    if route == "/fcls":
        path = artifact("abundance")
        names = "Not available"
        if path:
            try:
                import numpy as np
                if path.suffix == ".npz":
                    with np.load(path, allow_pickle=False) as values:
                        names = ", ".join(map(str, values["endmember_names"].tolist())) if "endmember_names" in values.files else "Names not stored"
                        dimensions = str(values["S"].shape) if "S" in values.files else "Shape not available"
                else:
                    dimensions = str(np.load(path, allow_pickle=False).shape)
            except (OSError, ValueError, KeyError):
                names = "Artifact could not be read"
        return shell("FCLS / Endmembers", f'<div class="eyebrow">Material interpretation</div><h1>FCLS / Endmembers</h1><p class="muted">Only generated abundance artifacts are displayed. FCLS is abundance estimation, not SR validation.</p><div class="card"><small>ENDMEMBER NAMES</small><strong>{html.escape(names)}</strong><small>ABUNDANCE DIMENSIONS</small><strong>{html.escape(locals().get("dimensions", "Not available"))}</strong></div><div class="card"><p class="muted">Abundance maps: {"Available" if path else "Not available"}; direct image display is used only for compatible 2-D maps.</p></div>', route)
    if route == "/about":
        flow = " ↓ ".join(("Sentinel-2 Input", "Preprocessing", "Bicubic Baseline", "SEN2SR", "Spectral Injection / Fusion", "Measurement Consistency", "Validation / Uncertainty", "FCLS / Endmember Analysis"))
        return shell("About", f'<div class="eyebrow">Project overview</div><h1>About SpectraGuard-SRM</h1><p class="muted">SpectraGuard-SRM is a Sentinel-2 spectral-spatial super-resolution and fusion system.</p><div class="card"><strong>{html.escape(flow)}</strong><p class="muted">The SR model is replaceable; SpectraGuard provides the analytical processing around the model, including fusion, consistency, validation, uncertainty, and material interpretation.</p></div>', route)
    raise KeyError(route)


def viewer(src: str, unavailable: str) -> str:
    return f'<div class="controls"><button onclick="zoom(-.15)">−</button><button onclick="zoom(.15)">+</button><button onclick="resetView()">Fit / reset</button></div><div class="viewer"><img id="scientific-image" src="{src}" alt="Scientific raster preview" onerror="this.remove();this.parentElement.innerHTML=\'<div class=empty>{unavailable}</div>\'"></div>'


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = unquote(urlparse(self.path).path)
        try:
            if route == "/scene.png":
                body, kind = raster_preview(config()[0]), "image/png"
            elif route.startswith("/result/") and route.endswith(".png"):
                role = route[8:-4]
                if role not in ("input", "bicubic", "sen2sr", "spectraguard") or artifact(role) is None:
                    raise FileNotFoundError()
                body, kind = _rgb_preview(role), "image/png"
            elif route.startswith("/map/") and route.endswith(".png"):
                role = route[5:-4]
                if role not in ("sam_map", "uncertainty") or artifact(role) is None:
                    raise FileNotFoundError()
                body, kind = _map_preview(role), "image/png"
            else:
                body, kind = page(route), "text/html; charset=utf-8"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
