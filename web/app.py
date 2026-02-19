"""FastAPI web application for constant curvature embeddings."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.load_data import VALID_DATASETS
from src.visualisation import project_to_2d
from web.config_manager import (
    get_default_config,
    load_config,
    save_config,
    validate_config,
)
from web.training_manager import TrainingManager

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(application):
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

training_manager = TrainingManager()

# Active WebSocket connections
_websocket_connections: set[WebSocket] = set()

# Color palette (tab10)
COLORS_HEX = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0


def _labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(labels), 3), dtype=np.float32)
    unique_labels = np.unique(labels)
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        colors[mask] = _hex_to_rgb(COLORS_HEX[idx % len(COLORS_HEX)])
    return colors


def _get_boundary_points(curvature: float) -> list[float] | None:
    if curvature == 0:
        return None
    theta = np.linspace(0, 2 * np.pi, 100)
    points = []
    for t in theta:
        points.extend([float(np.cos(t)), float(np.sin(t))])
    return points


async def _broadcast_json(data: dict):
    """Send JSON message to all connected WebSocket clients."""
    dead = set()
    for ws in _websocket_connections:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    _websocket_connections.difference_update(dead)


async def _broadcast_binary(data: bytes):
    """Send binary message to all connected WebSocket clients."""
    dead = set()
    for ws in _websocket_connections:
        try:
            await ws.send_bytes(data)
        except Exception:
            dead.add(ws)
    _websocket_connections.difference_update(dead)


def _training_update_callback(
    iteration: int, loss: float, model, phase: str, state
) -> None:
    """Called from training thread. Pushes update to WebSocket clients."""
    loop = _get_event_loop()
    if loop is None:
        return

    # Prepare JSON update
    json_msg = {
        "type": "update",
        "status": "running",
        "iteration": iteration,
        "max_iterations": state.max_iterations,
        "loss": loss,
        "phase": phase,
        "curvature": state.curvature,
    }

    # Prepare binary data: project embeddings to 2D, interleave with colors
    if state.embeddings is not None and state.labels is not None:
        k = state.curvature
        projection = "direct"  # default; spherical projection configured client-side
        x, y = project_to_2d(state.embeddings, k=k, i=0, j=1, projection=projection)
        colors = _labels_to_rgb(state.labels)
        # Interleave: [x0, y0, r0, g0, b0, x1, y1, r1, g1, b1, ...]
        n = len(x)
        binary = np.empty(n * 5, dtype=np.float32)
        binary[0::5] = x.astype(np.float32)
        binary[1::5] = y.astype(np.float32)
        binary[2::5] = colors[:, 0]
        binary[3::5] = colors[:, 1]
        binary[4::5] = colors[:, 2]
        binary_bytes = binary.tobytes()
    else:
        binary_bytes = None

    asyncio.run_coroutine_threadsafe(_broadcast_json(json_msg), loop)
    if binary_bytes:
        asyncio.run_coroutine_threadsafe(_broadcast_binary(binary_bytes), loop)


def _training_status_callback(status: str, error_message: str = "") -> None:
    """Called from training thread on status changes."""
    loop = _get_event_loop()
    if loop is None:
        return
    msg = {"type": "status", "status": status}
    if error_message:
        msg["message"] = error_message
    asyncio.run_coroutine_threadsafe(_broadcast_json(msg), loop)


_event_loop = None


def _get_event_loop():
    return _event_loop


# --- REST Endpoints ---


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
async def get_config():
    return load_config()


@app.post("/api/config")
async def post_config(config: dict):
    is_valid, error = validate_config(config)
    if not is_valid:
        return {"ok": False, "error": error}
    save_config(config)
    training_manager.clear_cache()
    return {"ok": True}


@app.post("/api/config/reset")
async def reset_config():
    default = get_default_config()
    save_config(default)
    training_manager.clear_cache()
    return {"ok": True, "config": default}


@app.post("/api/training/start")
async def start_training(config: dict):
    is_valid, error = validate_config(config)
    if not is_valid:
        return {"ok": False, "error": error}

    try:
        training_manager.start_training(
            config,
            update_callback=_training_update_callback,
            status_callback=_training_status_callback,
        )

        # Send boundary info to clients
        curvatures = config["experiments"]["curvatures"]
        curvature = curvatures[0] if curvatures else 0
        boundary = _get_boundary_points(curvature)
        await _broadcast_json({
            "type": "boundary",
            "points": boundary,
            "curvature": curvature,
        })

        return {"ok": True}
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/training/stop")
async def stop_training():
    training_manager.stop_training()
    return {"ok": True}


@app.get("/api/datasets")
async def get_datasets():
    return {"datasets": VALID_DATASETS}


# --- WebSocket ---


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _websocket_connections.add(ws)
    try:
        while True:
            # Keep connection alive, handle client messages if needed
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _websocket_connections.discard(ws)


if __name__ == "__main__":
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
    )
