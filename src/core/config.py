from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ProjectConfig:
    name: str


@dataclass
class AOIConfig:
    minx: float
    miny: float
    maxx: float
    maxy: float

    def as_bbox(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)


@dataclass
class DataConfig:
    bands: List[str]
    target_resolution_m: float
    working_resolution_m: float
    scene_path: str = ""
    aoi: Optional[AOIConfig] = None


@dataclass
class SRConfig:
    model: str
    scale: int
    tile_size: int
    overlap: int


@dataclass
class FusionConfig:
    epsilon: float
    alpha_min: float
    alpha_max: float


@dataclass
class ConsistencyConfig:
    iterations: int
    lambda_: float


@dataclass
class UnmixingConfig:
    solver: str
    endmembers: List[str]
    sensor_response: str = ""


@dataclass
class RunConfig:
    project: ProjectConfig
    data: DataConfig
    sr: SRConfig
    fusion: FusionConfig
    consistency: ConsistencyConfig
    unmixing: UnmixingConfig


def load_config(path: str) -> RunConfig:
    """Load a YAML configuration file into the shared RunConfig shape."""
    with Path(path).open("r", encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)

    return RunConfig(
        project=ProjectConfig(**values["project"]),
        data=DataConfig(
            bands=values["data"]["bands"],
            target_resolution_m=values["data"]["target_resolution_m"],
            working_resolution_m=values["data"]["working_resolution_m"],
            scene_path=values["data"].get("scene_path", ""),
            aoi=(
                AOIConfig(**values["data"]["aoi"])
                if values["data"].get("aoi") is not None
                else None
            ),
        ),
        sr=SRConfig(**values["sr"]),
        fusion=FusionConfig(
            epsilon=float(values["fusion"]["epsilon"]),
            alpha_min=values["fusion"]["alpha_min"],
            alpha_max=values["fusion"]["alpha_max"],
        ),
        consistency=ConsistencyConfig(
            iterations=values["consistency"]["iterations"],
            lambda_=values["consistency"]["lambda"],
        ),
        unmixing=UnmixingConfig(**values["unmixing"]),
    )
