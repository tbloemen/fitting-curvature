# Web Interface for Constant Curvature Embeddings

Interactive web application for visualizing t-SNE embeddings in constant curvature spaces.

## Quick Start

**Start the web server:**

```bash
uv run python -m web.app
```

**Access the interface:**
Open http://localhost:8080 in your browser

## Features

- **Interactive Configuration**: Edit all parameters through a form-based interface
- **Real-time Visualization**: Watch embeddings evolve during training via WebSocket
- **Multiple Geometries**: Support for hyperbolic (k < 0), Euclidean (k = 0), and spherical (k > 0) spaces
- **Projection Options**: Different visualization methods for spherical embeddings
- **Loss Tracking**: Real-time Plotly loss chart with log scale
- **Start/Stop Controls**: Pause training at any time

## Architecture

```
web/
├── app.py                    # FastAPI app: REST endpoints + WebSocket
├── training_manager.py       # Background training with callback push
├── config_manager.py         # Configuration load/save/validate
└── static/
    ├── index.html            # Single-page layout
    ├── style.css             # Styling
    ├── app.js                # Config form, WebSocket, UI state
    ├── threejs_plot.js       # Three.js 2D embedding visualization
    └── loss_chart.js         # Plotly loss chart
```

## How It Works

1. **Configuration**: All settings from `config.toml` are editable through the web UI
2. **Training**: Runs in a background thread, pushes updates via WebSocket
3. **Binary Protocol**: Embedding positions + colors sent as Float32Array for performance
4. **Visualization**: Three.js renders points, Plotly tracks loss history
5. **Thread Safety**: State updates are protected by locks

## Tips

- **Perplexity**: Use 5-50, with 30 being a good default
- **Curvature**:
  - k < 0 (hyperbolic): Good for hierarchical data
  - k = 0 (Euclidean): Standard t-SNE
  - k > 0 (spherical): Good for circular/periodic data
- **Projections**: For spherical embeddings, try different projections to find the best visualization

## Troubleshooting

**Import errors:**

- Make sure to run with `python -m web.app` from the project root
- Do NOT run with `python web/app.py` (will cause import errors)

**Port already in use:**

- Change the port in `web/app.py` (default is 8080)

**GPU memory issues:**

- Reduce `n_samples` or set device to CPU
