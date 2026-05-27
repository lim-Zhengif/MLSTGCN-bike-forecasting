# MLSTGCN Module Map

## Module Call Graph

```mermaid
flowchart TD
    A[train.py] --> B[datasets/air.py<br/>AirGraph]
    A --> C[datasets/air.py<br/>Air]
    A --> D[models/fusiongraph.py<br/>FusionGraphModel]
    A --> E[models/MSTGCN.py<br/>MSTGCN_submodule]
    A --> F[util.py<br/>masked metrics / scaler]

    B --> G[data/graph/pm25/*.npy]
    C --> H[data/temporal_data/pm25/*.npz]
    D --> I[data/SE/se_pm25.csv]

    D --> J[STAttBlock]
    J --> K[spatialAttention]
    J --> L[graphAttention]
    J --> M[gatedFusion]

    E --> N[MSTGCN_block x2]
    N --> O[cheb_conv]
    O --> D
    O --> P[torch_geometric Laplacian]

    A --> Q[PyTorch Lightning Trainer]
    Q --> R[train / val / test]
    R --> F
```

## Runtime Flow

1. `train.py` loads graph matrices and preprocessed PM2.5 samples.
2. `AirGraph` provides five candidate graphs: `dist`, `neighb`, `distri`, `tempp`, `func`.
3. `FusionGraphModel` fuses those graphs into one adjacency matrix for the current forward pass.
4. `MSTGCN_submodule` builds the graph Laplacian from that fused graph and runs spatial-temporal convolution.
5. `util.py` handles inverse scaling and masked MAE / MAPE / RMSE evaluation.

## Responsibilities By File

- `train.py`: experiment config, device selection, logger setup, Lightning training loop.
- `datasets/air.py`: PM2.5 dataset loading, graph selection, normalization.
- `models/fusiongraph.py`: multi-graph embedding, graph attention, fusion adjacency generation.
- `models/MSTGCN.py`: MSTGCN backbone using the fused graph.
- `util.py`: metrics, scaler, Lightning metric aggregation.
- `generate_training_data.py`: converts raw sequence data into sliding-window `npz` files.

## Data Shapes

- Input batch `x`: `[batch, hist_len, num_nodes, in_dim]`
- Label batch `y`: `[batch, pred_len, num_nodes, out_dim]`
- Fused graph: `[num_nodes, num_nodes]`
- Model output: `[batch, pred_len, num_nodes, out_dim]`

For the bundled PM2.5 data, the processed splits are:

- `train`: `(2186, 24, 92, 1)`
- `val`: `(728, 24, 92, 1)`
- `test`: `(729, 24, 92, 1)`
