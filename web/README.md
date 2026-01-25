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

- **Interactive Configuration**: Edit all parameters through a user-friendly interface
- **Real-time Visualization**: Watch embeddings evolve during training (updates every 10 iterations)
- **Multiple Geometries**: Support for hyperbolic (k < 0), Euclidean (k = 0), and spherical (k > 0) spaces
- **Projection Options**: Different visualization methods for spherical embeddings
- **Loss Tracking**: Real-time loss chart with log scale
- **Start/Stop Controls**: Pause training at any time

## Quick Test

For a fast test run:
1. Set `n_samples = 100` and `n_iterations = 100`
2. Select `curvature = 0` (Euclidean)
3. Click "Start Training"
4. Watch the embedding clusters form in real-time

## Production Run

For quality embeddings:
1. Set `n_samples = 1000-5000`
2. Set `n_iterations = 1000`
3. Configure perplexity (typically 30)
4. Select desired curvature
5. Click "Start Training"
6. Monitor convergence via loss chart

## Architecture

```
web/
├── app.py                    # Main NiceGUI application
├── training_manager.py       # Training execution with callbacks
├── config_manager.py         # Configuration management
└── components/
    ├── config_editor.py      # Configuration UI
    ├── training_control.py   # Start/stop controls
    └── plots.py              # Plotly visualizations
```

## How It Works

1. **Configuration**: All settings from `config.toml` are editable through the web UI
2. **Training**: Runs in a background thread using `ThreadPoolExecutor`
3. **Callbacks**: Training calls back every 10 iterations to update the UI
4. **Visualization**: Plotly charts auto-refresh to show current state
5. **Thread Safety**: State updates are protected by locks

## Tips

- **Perplexity**: Use 5-50, with 30 being a good default
- **Curvature**:
  - k < 0 (hyperbolic): Good for hierarchical data
  - k = 0 (Euclidean): Standard t-SNE
  - k > 0 (spherical): Good for circular/periodic data
- **Projections**: For spherical embeddings, try different projections to find the best visualization
- **Early Stopping**: Click "Stop Training" to halt before completion

## Troubleshooting

**Import errors:**
- Make sure to run with `python -m web.app` from the project root
- Do NOT run with `python web/app.py` (will cause import errors)

**Port already in use:**
- Change the port in `web/app.py` (default is 8080)

**GPU memory issues:**
- Reduce `n_samples` or set device to CPU

**Slow updates:**
- Visualization updates every 10 iterations
- Adjust the callback frequency in `src/embedding.py` if needed
