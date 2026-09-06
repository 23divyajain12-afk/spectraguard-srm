# SpectraGuard-SRM web UI

Run `python src/app/web_ui.py --port 8080` and open `http://127.0.0.1:8080`.

The local, read-only UI serves Home, Explore Scene, Process, and Results plus honest placeholders for later phases. It reads `config/default.yaml`, a configured Sentinel-2 SAFE or B04/B03/B02 paths, and existing `.npy`/`.npz` artifacts under `data/outputs/`. It never changes scientific data, configuration, or runs inference. Previews use display-only percentile stretching; scene rendering needs `rasterio` and `numpy`, while results need `numpy`.
