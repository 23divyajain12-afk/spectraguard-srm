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

The optional ESA OpenSR SEN2SR backend is selected with `sr.model: sen2sr`.
Install its optional stack with `python -m pip install -r requirements-sen2sr.txt`.
The adapter uses the official `opensr_model.SRLatentDiffusion` and
`load_pretrained()` checkpoint mechanism from
[ESAOpenSR/opensr-model](https://github.com/ESAOpenSR/opensr-model). It requires
four channels in B02/B03/B04/B08 order; B11 is not passed to SEN2SR. Use
`sr.device: auto` (CUDA when available, CPU otherwise), `sr.checkpoint`, and
`sr.sampling_steps` in configuration. Missing dependencies or checkpoints fail
explicitly; no fallback to the baseline occurs.

The dashboard helpers in `src/app/dashboard.py` read the saved NPZ/JSON
artifacts from `data/outputs/`.

Validation metrics compare the degraded fused product with the original 10m
Sentinel-2 observation and are reported as internal consistency measurements,
not 2.5m ground-truth accuracy. A future validation workflow should add an
independent high-resolution reference or Wald-protocol assessment; the main
pipeline intentionally does not require either dataset.
