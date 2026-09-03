import numpy as np
import warnings

from src.core.config import RunConfig
from src.core.schemas import SRModel


class NumpyInterpolationModel:
    """Small dependency-free SR adapter used when no external model is configured."""

    def __init__(self, scale: int, model_name: str) -> None:
        if scale < 1:
            raise ValueError("SR scale must be at least 1")
        self.scale = scale
        self.model_name = model_name

    def predict(self, spatial_input: np.ndarray) -> np.ndarray:
        array = np.asarray(spatial_input, dtype=np.float32)
        if array.ndim not in (2, 3):
            raise ValueError("spatial_input must have shape (H,W) or (C,H,W)")
        if array.shape[-2] == 0 or array.shape[-1] == 0:
            raise ValueError("spatial_input cannot be empty")
        return _resize(array, array.shape[-2] * self.scale, array.shape[-1] * self.scale)


class Sen2SRModel:
    """Adapter for ESA OpenSR's pretrained four-channel SEN2SR model."""

    def __init__(self, config: RunConfig) -> None:
        try:
            import torch
            import opensr_model
        except ImportError as error:
            raise ImportError(
                "SEN2SR requires the optional 'opensr-model' and 'torch' packages. "
                "Install them before selecting sr.model='sen2sr'."
            ) from error

        self._torch = torch
        requested_device = config.sr.device.lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA is unavailable; SEN2SR inference will run on CPU.", RuntimeWarning)
            requested_device = "cpu"
        if requested_device not in {"cpu", "cuda"}:
            raise ValueError("sr.device must be 'auto', 'cpu', or 'cuda'")
        self.device = requested_device
        try:
            from omegaconf import OmegaConf
            from io import StringIO
            import requests
            response = requests.get(
                "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/main/"
                "opensr_model/configs/config_10m.yaml",
                timeout=30,
            )
            response.raise_for_status()
            model_config = OmegaConf.load(StringIO(response.text))
        except (ImportError, OSError, ValueError, AttributeError) as error:
            raise RuntimeError(
                "SEN2SR configuration could not be loaded. Install opensr-model "
                "dependencies and provide network access to the official config."
            ) from error
        self.model = opensr_model.SRLatentDiffusion(model_config, device=self.device)
        checkpoint = config.sr.checkpoint or getattr(model_config, "ckpt_version", "")
        if not checkpoint:
            raise ValueError(
                "SEN2SR checkpoint is not configured. Set sr.checkpoint to an "
                "official checkpoint/version."
            )
        try:
            self.model.load_pretrained(checkpoint)
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"SEN2SR checkpoint '{checkpoint}' could not be loaded; "
                "obtain it through the official opensr-model mechanism."
            ) from error
        self.model_name = "SEN2SR"
        self.model_version = str(checkpoint)
        self.scale = config.sr.scale
        self.sampling_steps = config.sr.sampling_steps

    def predict(self, spatial_input: np.ndarray) -> np.ndarray:
        array = np.asarray(spatial_input, dtype=np.float32)
        if array.ndim != 3 or array.shape[0] != 4:
            raise ValueError("SEN2SR requires exactly four channels: B02, B03, B04, B08")
        tensor = self._torch.from_numpy(array).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            output = self.model.forward(tensor, sampling_steps=self.sampling_steps)
        result = output.detach().cpu().numpy()
        if result.ndim == 4:
            result = result[0]
        if result.shape != (4, array.shape[1] * 4, array.shape[2] * 4):
            raise ValueError("SEN2SR returned an unexpected output shape")
        return np.clip(result, 0.0, 1.0).astype(np.float32)


def _resize(array: np.ndarray, height: int, width: int) -> np.ndarray:
    channels = array[None, ...] if array.ndim == 2 else array
    y = np.linspace(0, channels.shape[1] - 1, height)
    x = np.linspace(0, channels.shape[2] - 1, width)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, channels.shape[1] - 1)
    x1 = np.minimum(x0 + 1, channels.shape[2] - 1)
    wy = (y - y0)[:, None]
    wx = (x - x0)[None, :]
    top = channels[:, y0[:, None], x0[None, :]] * (1 - wx)
    top += channels[:, y0[:, None], x1[None, :]] * wx
    bottom = channels[:, y1[:, None], x0[None, :]] * (1 - wx)
    bottom += channels[:, y1[:, None], x1[None, :]] * wx
    result = (top * (1 - wy) + bottom * wy).astype(np.float32)
    return result[0] if array.ndim == 2 else result


def load_sr_model(config: RunConfig) -> SRModel:
    """Load the configured replaceable SR model adapter."""
    if not config.sr.model:
        raise ValueError("SR model name must not be empty")
    if config.sr.model == "sen2sr":
        if config.sr.scale != 4:
            raise ValueError("SEN2SR requires sr.scale=4")
        return Sen2SRModel(config)
    if config.sr.model not in {"numpy_baseline", "numpy_interpolation"}:
        raise ValueError(
            f"SR model backend '{config.sr.model}' is unavailable; "
            "add its adapter and weights, or select 'numpy_baseline'"
        )
    return NumpyInterpolationModel(config.sr.scale, config.sr.model)
