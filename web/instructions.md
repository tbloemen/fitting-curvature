### How to Use

1. **Configure Settings**: Use the left panel to adjust embedding parameters
   - Select dataset and number of samples
   - Choose embedding dimension (typically 2 for visualization)
   - Set number of iterations
   - Configure perplexity and momentum parameters
   - Select a curvature value (k < 0 for hyperbolic, k = 0 for Euclidean, k > 0 for spherical)

2. **Save Configuration**: Click "Save Configuration" to persist your settings

3. **Start Training**: Click "Start Training" to begin the embedding process
   - The visualization updates every 10 iterations
   - Progress bar shows current iteration
   - Loss chart tracks convergence

4. **Monitor Progress**:
   - Watch the embedding plot evolve as clusters form
   - Check the loss chart to verify convergence
   - Phase indicator shows "Early Exaggeration" or "Main Phase"

5. **Stop Training**: Click "Stop Training" to halt the process early if needed

### Tips

- **Quick Test**: Use n_samples=100 and n_iterations=100 for fast results
- **Production**: Use n_samples=1000-5000 and n_iterations=1000 for quality embeddings
- **Perplexity**: Typical values are 5-50, with 30 being a good default
- **Curvature**: Start with k=0 (Euclidean), then try k=-1 (hyperbolic) or k=1 (spherical)
- **Projection**: For spherical embeddings (k > 0), try different projections to see which visualizes best

### Geometry Types

- **Hyperbolic (k < 0)**: Good for hierarchical data, displayed in Poincaré disk
- **Euclidean (k = 0)**: Standard t-SNE, no curvature
- **Spherical (k > 0)**: Good for circular/periodic data, displayed on sphere surface
