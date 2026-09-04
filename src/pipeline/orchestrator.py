import json
import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.consistency.projection import project_to_measurement_consistency
from src.core.config import RunConfig, load_config
from src.core.schemas import EndmemberSet, FusedCube, RawSpectrum, RunMetadata, SentinelCube
from src.data.geospatial import crop_to_aoi, reproject_match
from src.data.preprocessing import preprocess
from src.data.sentinel_io import load_sentinel
from src.fusion.bicubic import bicubic_upscale
from src.fusion.detail_residual import extract_residual
from src.fusion.spectral_injection import fuse_multispectral, run_safety_checks
from src.library.endmember_preparation import prepare_endmembers
from src.library.sensor_response import load_sensor_response
from src.library.usgs_io import load_usgs_spectrum
from src.sr.model_adapter import load_sr_model
from src.sr.spatial_representation import build_spatial_input
from src.sr.tiling import tiled_inference
from src.unmixing.fcls import fcls
from src.validation.metrics import evaluate
from src.validation.uncertainty import estimate_uncertainty


LOGGER = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "outputs"


def _load_fixture_endmembers(path: Path) -> EndmemberSet:
    with np.load(path, allow_pickle=False) as values:
        required = {"endmember_A", "endmember_names", "material_classes", "source_spectra"}
        missing = required.difference(values.files)
        if missing:
            raise ValueError(f"Fixture is missing endmember fields: {sorted(missing)}")
        return EndmemberSet(
            A=np.asarray(values["endmember_A"], dtype=np.float32),
            names=[str(value) for value in values["endmember_names"].tolist()],
            material_classes=[str(value) for value in values["material_classes"].tolist()],
            source_spectra=[str(value) for value in values["source_spectra"].tolist()],
        )


def _load_endmembers(
    config: RunConfig, scene_path: Path
) -> Optional[EndmemberSet]:
    if config.unmixing.sensor_response and not config.unmixing.endmembers:
        raise ValueError(
            "unmixing.endmembers must be configured with unmixing.sensor_response"
        )
    if config.unmixing.endmembers:
        if not config.unmixing.sensor_response:
            raise ValueError(
                "unmixing.sensor_response must be configured with spectral endmembers"
            )
        spectra: list[RawSpectrum] = [
            load_usgs_spectrum(path) for path in config.unmixing.endmembers
        ]
        response = load_sensor_response(
            config.unmixing.sensor_response,
            list(config.data.bands) or ["B02", "B03", "B04", "B08", "B11"],
        )
        return prepare_endmembers(spectra, response, config)
    if scene_path.suffix.lower() == ".npz":
        return _load_fixture_endmembers(scene_path)
    return None


def _configured_scene_path(config: RunConfig) -> Path:
    if not config.data.scene_path:
        raise ValueError("data.scene_path must identify a Sentinel scene")
    scene_path = Path(config.data.scene_path)
    if not scene_path.is_absolute():
        scene_path = (Path.cwd() / scene_path).resolve()
    if not scene_path.is_file() and not scene_path.is_dir():
        raise FileNotFoundError(f"Configured Sentinel scene does not exist: {scene_path}")
    return scene_path


def _align_configured_geometry(cube: SentinelCube) -> SentinelCube:
    """Apply optional geometry metadata without adding configuration fields."""
    target_crs = cube.meta.get("target_crs")
    target_transform = cube.meta.get("target_transform")
    if target_crs is not None or target_transform is not None:
        if target_crs is None or target_transform is None:
            raise ValueError("target_crs and target_transform must be provided together")
        cube = reproject_match(cube, str(target_crs), tuple(target_transform))
    aoi = cube.meta.get("aoi")
    if aoi is not None:
        cube = crop_to_aoi(cube, aoi)
    return cube


def _save_cube(cube: SentinelCube, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        data=cube.data,
        band_names=np.asarray(cube.band_names),
        crs=np.asarray(cube.crs),
        transform=np.asarray(cube.transform),
        resolution_m=np.asarray(cube.resolution_m),
        bounds=np.asarray(cube.bounds),
        mask=cube.mask,
        nodata=np.asarray(cube.nodata if cube.nodata is not None else np.nan),
        acquisition_time=np.asarray(cube.acquisition_time or ""),
        meta_json=np.asarray(json.dumps(cube.meta, default=str)),
        alpha_map=np.asarray(getattr(cube, "alpha_map", None))
        if getattr(cube, "alpha_map", None) is not None
        else np.asarray([]),
        provenance=np.asarray(getattr(cube, "provenance", "")),
        anomaly_mask=np.asarray(getattr(cube, "anomaly_mask", None))
        if getattr(cube, "anomaly_mask", None) is not None
        else np.asarray([]),
    )


def _save_metadata(metadata: RunMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.__dict__, indent=2, default=str), encoding="utf-8")


def _make_fused_cube(cube: FusedCube, provenance: str) -> FusedCube:
    return FusedCube(
        data=cube.data,
        band_names=list(cube.band_names),
        crs=cube.crs,
        transform=cube.transform,
        resolution_m=cube.resolution_m,
        bounds=cube.bounds,
        mask=cube.mask,
        nodata=cube.nodata,
        acquisition_time=cube.acquisition_time,
        meta=dict(cube.meta),
        alpha_map=cube.alpha_map,
        provenance=provenance,
        anomaly_mask=cube.anomaly_mask,
    )


def _select_rgb_nir_cube(cube: SentinelCube) -> SentinelCube:
    required = ["B02", "B03", "B04", "B08"]
    missing = [band for band in required if band not in cube.band_names]
    if missing:
        raise ValueError(f"SEN2SR source cube is missing bands: {missing}")
    indices = [cube.band_names.index(band) for band in required]
    return SentinelCube(
        data=cube.data[indices].copy(),
        band_names=required,
        crs=cube.crs,
        transform=cube.transform,
        resolution_m=cube.resolution_m,
        bounds=cube.bounds,
        mask=cube.mask.copy(),
        nodata=cube.nodata,
        acquisition_time=cube.acquisition_time,
        meta={**cube.meta, "source_bands": required, "b11_processed_by_sen2sr": False},
    )


def _run_pipeline(
    config: RunConfig,
    scene_path: Path,
    endmembers: Optional[EndmemberSet],
    output_root: Path,
    fcls_sample_limit: Optional[int] = None,
) -> RunMetadata:
    LOGGER.info("[1/9] Loading Sentinel-2")
    aoi = config.data.aoi.as_bbox() if config.data.aoi is not None else None
    original = load_sentinel(str(scene_path), config, aoi_bbox=aoi)
    original = _align_configured_geometry(original)
    LOGGER.info("[2/9] Preprocessing")
    original = preprocess(original, config)
    model_input_cube = (
        _select_rgb_nir_cube(original)
        if config.sr.model == "sen2sr"
        else original
    )
    LOGGER.info("[3/9] Bicubic baseline")
    bicubic = bicubic_upscale(model_input_cube, config.sr.scale)

    spatial_input = build_spatial_input(original, config)
    LOGGER.info("[4/9] SEN2SR inference" if config.sr.model == "sen2sr"
                else "[4/9] SR inference")
    model = load_sr_model(config)
    spatial_hr = tiled_inference(
        model, spatial_input.array, config.sr.tile_size, config.sr.overlap
    )
    if config.sr.model == "sen2sr":
        residual = extract_residual(spatial_hr, bicubic.data)
        LOGGER.info(
            "[5/9] Spectral fusion (preserving four-channel SEN2SR RGB-NIR; "
            "B11 not processed by SEN2SR)"
        )
        fused = FusedCube(
            data=spatial_hr,
            band_names=list(model_input_cube.band_names),
            crs=bicubic.crs,
            transform=bicubic.transform,
            resolution_m=bicubic.resolution_m,
            bounds=bicubic.bounds,
            mask=bicubic.mask.copy(),
            nodata=bicubic.nodata,
            acquisition_time=bicubic.acquisition_time,
            meta={
                **bicubic.meta,
                "source_bands": list(model_input_cube.band_names),
                "b11_processed_by_sen2sr": False,
            },
            alpha_map=None,
            provenance="sen2sr_rgb_nir",
            anomaly_mask=None,
        )
        fused = run_safety_checks(fused, config)
        validation_cube = model_input_cube
    else:
        spatial_base = np.mean(bicubic.data[: spatial_hr.shape[0]], axis=0)
        spatial_hr_channel = np.mean(spatial_hr, axis=0)
        residual = extract_residual(spatial_hr_channel, spatial_base)
        LOGGER.info("[5/9] Spectral fusion")
        fused = fuse_multispectral(bicubic, spatial_base, spatial_hr_channel, config)
        fused = run_safety_checks(fused, config)
        fused = _make_fused_cube(
            fused,
            f"{getattr(model, 'model_name', config.sr.model)}+{fused.provenance}",
        )
        validation_cube = original
    LOGGER.info("[6/9] Measurement consistency")
    fused = project_to_measurement_consistency(fused, validation_cube, config)
    LOGGER.info("[7/9] Validation")
    metrics = evaluate(fused, validation_cube, None, config)
    LOGGER.info("[8/9] Uncertainty")
    uncertainty = estimate_uncertainty(fused, metrics, config)

    sample_count = None
    abundance = None
    if endmembers is not None:
        LOGGER.info("[9/9] FCLS unmixing")
        fcls_cube = fused
        if fcls_sample_limit is not None:
            valid_rows, valid_cols = np.where(fused.mask)
            sample_count = min(fcls_sample_limit, len(valid_rows))
            sample_mask = np.zeros_like(fused.mask, dtype=bool)
            sample_mask[valid_rows[:sample_count], valid_cols[:sample_count]] = True
            fcls_cube = replace(fused, mask=sample_mask)
        abundance = fcls(fcls_cube, endmembers, config)
    else:
        LOGGER.info(
            "[9/9] FCLS unmixing / skipped: no endmember/sensor-response "
            "configuration supplied."
        )

    _save_cube(original, output_root / "validation" / "original.npz")
    _save_cube(bicubic, output_root / "bicubic" / "bicubic.npz")
    _save_cube(fused, output_root / "fused" / "fused.npz")
    output_root.joinpath("sr").mkdir(parents=True, exist_ok=True)
    np.savez(
        output_root / "sr" / "spatial_prediction.npz",
        data=spatial_hr,
        transform=np.asarray(bicubic.transform),
        crs=np.asarray(bicubic.crs),
        bounds=np.asarray(bicubic.bounds),
        resolution_m=np.asarray(bicubic.resolution_m),
        mask=bicubic.mask,
        source_bands=np.asarray(spatial_input.source_bands),
        method=np.asarray(spatial_input.method),
        model=np.asarray(getattr(model, "model_name", config.sr.model)),
        model_version=np.asarray(getattr(model, "model_version", "baseline")),
        device=np.asarray(getattr(model, "device", "cpu")),
        b11_processed_by_sen2sr=np.asarray(False),
    )
    output_root.joinpath("validation").mkdir(parents=True, exist_ok=True)
    output_root.joinpath("uncertainty").mkdir(parents=True, exist_ok=True)
    output_root.joinpath("abundance").mkdir(parents=True, exist_ok=True)
    np.savez(
        output_root / "validation" / "metrics.npz",
        sam_map=metrics.sam_map,
        per_band_rmse=np.asarray(list(metrics.per_band_rmse.values())),
        per_band_mae=np.asarray(list(metrics.per_band_mae.values())),
        band_names=np.asarray(list(metrics.per_band_rmse)),
        sam_mean=np.asarray(metrics.sam_mean),
        reference_type=np.asarray(metrics.reference_type),
    )
    (output_root / "validation" / "metrics.json").write_text(
        json.dumps(
            {
                "sam_mean": metrics.sam_mean,
                "per_band_rmse": metrics.per_band_rmse,
                "per_band_mae": metrics.per_band_mae,
                "psnr": metrics.psnr,
                "ssim": metrics.ssim,
                "ergas": metrics.ergas,
                "reference_type": metrics.reference_type,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez(
        output_root / "uncertainty" / "uncertainty.npz",
        U=uncertainty.U,
        sam=uncertainty.components["sam"],
        reconstruction=uncertainty.components["reconstruction"],
        detail=uncertainty.components["detail"],
        weights_json=np.asarray(json.dumps(uncertainty.weights)),
        label=np.asarray(uncertainty.label),
    )
    if abundance is not None:
        abundance_name = (
            "abundance_sample.npz" if sample_count is not None else "abundance.npz"
        )
        np.savez(
            output_root / "abundance" / abundance_name,
            S=abundance.S,
            residual=np.asarray(abundance.residual),
            endmember_names=np.asarray(abundance.endmember_names),
            solver=np.asarray(abundance.solver),
            sample_count=np.asarray(sample_count if sample_count is not None else -1),
        )

    metadata = RunMetadata(
        scene_id=str(original.meta.get("scene_id", "unknown")),
        aoi={"bounds": original.bounds, "crs": original.crs},
        bands=list(fused.band_names),
        model_name=str(getattr(model, "model_name", config.sr.model)),
        model_version=str(getattr(model, "model_version", "baseline")),
        scale=config.sr.scale,
        fusion_params={"epsilon": config.fusion.epsilon, "alpha_min": config.fusion.alpha_min, "alpha_max": config.fusion.alpha_max},
        consistency_params={"iterations": config.consistency.iterations, "lambda": config.consistency.lambda_},
        endmember_set=(
            "fixture" if fcls_sample_limit is not None
            else "configured" if endmembers is not None
            else "not_configured"
        ),
        validation_settings={
            "reference_type": metrics.reference_type,
            "fcls_result": "sample" if sample_count is not None else "full",
            "fcls_sample_count": sample_count,
            "unmixing_skipped": endmembers is None,
            "source_bands": list(model_input_cube.band_names),
            "b11_processed_by_sen2sr": False,
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
        software_versions={"python": sys.version.split()[0], "numpy": np.__version__},
    )
    _save_metadata(metadata, output_root / "reports" / "run_metadata.json")
    LOGGER.info("Artifact writing complete: %s", output_root)
    return metadata


def run_pipeline(config: RunConfig) -> RunMetadata:
    """Run using the scene path configured under ``data.scene_path``."""
    if not isinstance(config, RunConfig):
        raise TypeError("config must be a RunConfig")
    scene_path = _configured_scene_path(config)
    endmembers = _load_endmembers(config, scene_path)
    return _run_pipeline(config, scene_path, endmembers, OUTPUT_ROOT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    configuration_path = REPOSITORY_ROOT / "config" / "default.yaml"
    configuration = load_config(str(configuration_path))
    scene_path = _configured_scene_path(configuration)
    _run_pipeline(
        configuration,
        scene_path,
        _load_endmembers(configuration, scene_path),
        OUTPUT_ROOT,
        fcls_sample_limit=256,
    )
