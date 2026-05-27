# Using Your Own Data With FusionGraph

## Goal

This checklist explains what you need to prepare if you want to keep the multi-graph fusion idea in `models/fusiongraph.py` but replace the bundled PM2.5 dataset with your own task.

## 1. Define Your Prediction Task

Before touching the code, make these choices explicit:

- What is one node
  - Example: a sensor, a station, a region, a road segment, or a business zone
- What is one time step
  - Example: 5 minutes, 1 hour, 1 day
- What are you predicting
  - Example: traffic flow, PM2.5, demand, temperature
- How many historical steps you feed in
  - Current project default: `hist_len = 24`
- How many future steps you predict
  - Current project default: `pred_len = 24`

## 2. Prepare Temporal Data

Your temporal data must finally become:

- `x`: `[num_samples, hist_len, num_nodes, in_dim]`
- `y`: `[num_samples, pred_len, num_nodes, out_dim]`

For a single-variable task, `in_dim = out_dim = 1`.

You need three split files:

- `data/temporal_data/<your_task>/train.npz`
- `data/temporal_data/<your_task>/val.npz`
- `data/temporal_data/<your_task>/test.npz`

Each `npz` file should contain at least:

- `x`
- `y`

Recommended checks:

- Node order is consistent across all splits
- No mismatch between temporal data node count and graph node count
- Missing values are handled before training or masked consistently
- Train/val/test are split by time, not randomly shuffled time points

## 3. Prepare Multi-Graph Inputs

FusionGraph expects several adjacency-like matrices with the same node set.

Each graph must have shape:

- `[num_nodes, num_nodes]`

In the current project, the five graphs are:

- `dist`: distance graph
- `neighb`: neighborhood or connectivity graph
- `distri`: distribution similarity graph
- `tempp`: temporal pattern similarity graph
- `func`: functional similarity graph

You do not have to keep exactly the same semantics, but if you want to reuse the current code with minimal edits, it is easiest to still provide five matrices under these names.

Recommended file layout:

- `data/graph/<your_task>/dist.npy`
- `data/graph/<your_task>/neigh.npy`
- `data/graph/<your_task>/func.npy`
- `data/graph/<your_task>/<your_task>_distri_kl.npy`
- `data/graph/<your_task>/<your_task>_distri_ws.npy`
- `data/graph/<your_task>/tempp_<your_task>.npy`

Practical guidance for each graph:

- Distance graph
  - Derived from physical distance, network distance, or travel time
- Neighborhood graph
  - Binary adjacency or local k-nearest-neighbor graph
- Functional graph
  - Similarity based on land use, POI, category, business type, or role
- Distribution graph
  - Similarity between node-level empirical distributions
- Temporal pattern graph
  - Similarity between node-level daily or weekly temporal curves

Recommended checks:

- Diagonal handling is intentional
- Matrix values are nonnegative if that is assumed by your formulation
- All graphs use the same node ordering
- All graphs are aligned to the exact same node set as the temporal data

## 4. Prepare Spatial Embedding

The fusion module also uses a spatial embedding file:

- `data/SE/se_<your_task>.csv`

Expected shape:

- `[num_nodes, D]`

In this project:

- `D = M * d`
- default `M = 24`
- default `d = 6`
- so default `D = 144`

You can generate this embedding using methods like:

- Node2Vec on one graph
- Node2Vec on a merged graph
- Another graph embedding method with the same output dimension

Recommended checks:

- Row count equals `num_nodes`
- Row order matches the node order used in temporal data and graphs
- Embedding dimension matches `M * d`

## 5. Code Changes You Will Usually Need

To plug in a new dataset cleanly, you will usually need to adapt:

- `train.py`
  - add your task name
  - point to `data/temporal_data/<your_task>`
  - point to `data/graph/<your_task>`
- `datasets/air.py`
  - generalize or duplicate this loader for your dataset
  - adjust graph filenames if needed
- `models/fusiongraph.py`
  - add `se_<your_task>.csv` path resolution
- `generate_training_data.py`
  - only if you want to build `train/val/test.npz` from raw data

If your task still uses one variable and the same five-graph structure, you usually do not need to change the MSTGCN backbone itself.

## 6. Minimal Compatibility Conditions

These conditions must all hold at once:

- Temporal data node count = graph node count = spatial embedding row count
- Temporal data node order = graph node order = spatial embedding row order
- Spatial embedding dimension = `M * d`
- Graph filenames match what the loader expects
- Data split files exist before training starts

## 7. Suggested Workflow

1. Freeze a node list and node ordering.
2. Build raw multivariate time series table with shape `[time, num_nodes]` or equivalent.
3. Generate `train.npz`, `val.npz`, `test.npz`.
4. Build the five graph matrices on the same node ordering.
5. Generate `se_<your_task>.csv`.
6. Update loader paths and task name handling.
7. Run a 1-epoch smoke test.
8. Run the full training.

## 8. Common Failure Modes

- `FileNotFoundError`
  - path is wrong or task-specific file names do not match loader expectations
- shape mismatch in fusiongraph
  - graph size and temporal node count are inconsistent
- spatial embedding error
  - `se_<your_task>.csv` row count or dimension is wrong
- poor results even though code runs
  - graph semantics are weak, node alignment is wrong, or train/val/test split leaks time information
