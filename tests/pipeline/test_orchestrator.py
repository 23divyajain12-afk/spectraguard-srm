from dataclasses import replace
from pathlib import Path

import pytest

from src.core.config import load_config
from src.pipeline import orchestrator


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "aoi_small.npz"


def test_run_pipeline_uses_configured_scene_path(monkeypatch):
    config = load_config(str(REPOSITORY_ROOT / "config" / "default.yaml"))
    configured = replace(config.data, scene_path=str(FIXTURE))
    config = replace(config, data=configured)
    expected = object()
    captured = {}

    def fake_run_pipeline(config, scene_path, endmembers, output_root):
        captured["path"] = scene_path
        return expected

    monkeypatch.setattr(orchestrator, "_run_pipeline", fake_run_pipeline)
    assert orchestrator.run_pipeline(config) is expected
    assert captured["path"] == FIXTURE.resolve()


def test_run_pipeline_rejects_missing_scene_path():
    config = load_config(str(REPOSITORY_ROOT / "config" / "default.yaml"))
    config = replace(config, data=replace(config.data, scene_path="missing-scene.npz"))

    with pytest.raises(FileNotFoundError, match="Configured Sentinel scene"):
        orchestrator.run_pipeline(config)


def test_fixture_smoke_entrypoint_runs():
    config = load_config(str(REPOSITORY_ROOT / "config" / "default.yaml"))
    assert config.data.scene_path == "tests/fixtures/aoi_small.npz"


def test_safe_without_unmixing_configuration_skips_unmixing(monkeypatch, tmp_path):
    safe_path = tmp_path / "scene.SAFE"
    safe_path.mkdir()
    config = load_config(str(REPOSITORY_ROOT / "config" / "default.yaml"))
    config = replace(config, data=replace(config.data, scene_path=str(safe_path)))
    captured = {}

    def fake_run_pipeline(config, scene_path, endmembers, output_root):
        captured["endmembers"] = endmembers
        return object()

    monkeypatch.setattr(orchestrator, "_run_pipeline", fake_run_pipeline)
    orchestrator.run_pipeline(config)

    assert captured["endmembers"] is None
