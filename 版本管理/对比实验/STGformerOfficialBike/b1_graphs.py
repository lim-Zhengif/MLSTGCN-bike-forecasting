"""Load existing, node-aligned relation artifacts; never build from holdout."""
from pathlib import Path
import numpy as np
import pandas as pd
from protocol import audit_graph_contract, row_normalize, sha256_file

RELATION_FILES = {"dist": "dist.npy", "distri": "bike_heuristic.npy"}
RELATION_FILES.update({"od%02d" % hour: "od%02d.npy" % hour for hour in range(0, 24, 3)})


def load_relations(graph_dir, num_nodes):
    graph_dir = Path(graph_dir)
    mapping = pd.read_csv(graph_dir / "selected_node_mapping.csv")
    if not np.array_equal(mapping["Node_ID"].to_numpy(), np.arange(num_nodes)):
        raise ValueError("Mapping rows must be contiguous Node_ID order 0..N-1")
    audit = audit_graph_contract(graph_dir, "dist", num_nodes)
    matrices, sources = [], {}
    for name, filename in RELATION_FILES.items():
        path = graph_dir / filename
        raw = np.load(path)
        if raw.shape != (num_nodes, num_nodes) or not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError("Invalid relation graph: %s" % path)
        matrices.append(row_normalize(raw, add_self=True))
        sources[name] = {"filename": filename, "sha256": sha256_file(path)}
    audit.update(model_graph_source="adaptive_plus_static_multirelation",
                 b0_graph_role="active_relation", relations=sources,
                 normalization="add_identity_then_row_normalize",
                 orientation="row_i_aggregates_column_j; preserves stored OD orientation",
                 anchor_conditioning=False)
    return np.stack(matrices), list(RELATION_FILES), audit
