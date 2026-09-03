# SpectraGuard-SRM

## Install and test

```powershell
python -m pip install -r requirements.txt
pytest -q
```

## Fixture pipeline

The default configuration uses the deterministic fixture and the explicitly
labelled `numpy_baseline` interpolation model:

```powershell
python -m src.pipeline.orchestrator
```

Outputs are written below `data/outputs/`. Validation metrics are available as
`validation/metrics.npz` and `validation/metrics.json`; without reference
imagery they are internal-consistency metrics, not ground truth.

## Real Sentinel-2 processing

Set `data.scene_path` to a local Sentinel-2 L2A `.SAFE` directory and optionally
set `data.aoi` to CRS-coordinate bounds (`minx`, `miny`, `maxx`, `maxy`).
The loader reads B02, B03, B04, B08 at 10 m and B11 at 20 m, windowing and
reprojecting the requested area. No Sentinel or model data is downloaded.

For real unmixing, configure `unmixing.endmembers` with local USGS-compatible
two-column spectrum files and `unmixing.sensor_response` with a local sensor
response CSV/NPZ. Missing scientific inputs fail explicitly. A future ML model
must implement `SRModel.predict()` and be registered in
`src/sr/model_adapter.py`; weights are supplied locally and are never
downloaded automatically.

The dashboard helpers in `src/app/dashboard.py` read the saved NPZ/JSON
artifacts from `data/outputs/`.
