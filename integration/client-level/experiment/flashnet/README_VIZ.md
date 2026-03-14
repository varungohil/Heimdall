# Heimdall Visualization

## Running the visualization

The visualization must be served over HTTP (not opened via `file://`) for the model dropdown to work:

```bash
cd integration/client-level/experiment/flashnet
python -m http.server 8000
# Open http://localhost:8000/heimdall_viz.html
```

## Exporting models to JSON

To convert all `model.keras` files under the data folder to JSON weights:

```bash
python integration/client-level/experiment/flashnet/training/export_all_models_for_viz.py \
  --data-dir integration/client-level/data \
  --output-dir integration/client-level/experiment/flashnet/weights
```

This creates:
- `weights/weights_000.json` ... `weights_011.json` (one per model)
- `weights/manifest.json` (list of models for the dropdown)

## Single model export

To export a single model:

```bash
python integration/client-level/experiment/flashnet/training/export_weights_for_viz.py \
  -model path/to/model.keras \
  -dataset path/to/mldrive0.csv \
  -output custom_weights.json
```
