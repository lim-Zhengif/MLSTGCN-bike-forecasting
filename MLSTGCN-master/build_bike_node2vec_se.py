import argparse
import json
import os

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def detect_project_root(start_dir):
    current = os.path.abspath(start_dir)
    while True:
        if (
            os.path.isdir(os.path.join(current, 'data'))
            and os.path.isdir(os.path.join(current, 'models'))
            and os.path.isdir(os.path.join(current, 'datasets'))
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.abspath(start_dir)
        current = parent


PROJECT_ROOT = detect_project_root(SCRIPT_DIR)


DEFAULT_GRAPH_FILES = [
    'dist.npy',
    'neigh.npy',
    'func.npy',
    'bike_heuristic.npy',
    'tempp_bike.npy',
]


def parse_graph_weights(raw_value, graph_count):
    if raw_value is None:
        return np.ones(graph_count, dtype=np.float32)
    parts = [item.strip() for item in raw_value.split(',') if item.strip()]
    if len(parts) != graph_count:
        raise ValueError('graph_weights length must match graph count: %s != %s' % (len(parts), graph_count))
    weights = np.asarray([float(item) for item in parts], dtype=np.float32)
    if np.any(weights < 0):
        raise ValueError('graph_weights must be non-negative')
    if np.allclose(weights.sum(), 0):
        raise ValueError('graph_weights cannot all be zero')
    return weights


def normalize_graph(graph, method):
    graph = np.asarray(graph, dtype=np.float32).copy()
    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph = np.maximum(graph, 0.0)
    np.fill_diagonal(graph, 0.0)

    if method == 'none':
        return graph

    if method == 'max':
        scale = float(graph.max())
        if scale > 0:
            graph /= scale
        return graph

    if method == 'row':
        row_sum = graph.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return graph / row_sum

    raise ValueError('Unsupported normalize method: %s' % method)


def sparsify_topk(graph, top_k):
    if top_k <= 0 or top_k >= graph.shape[0]:
        return graph

    sparse_graph = np.zeros_like(graph)
    for idx in range(graph.shape[0]):
        row = graph[idx].copy()
        row[idx] = 0.0
        positive = np.where(row > 0)[0]
        if len(positive) == 0:
            continue
        if len(positive) <= top_k:
            keep_idx = positive
        else:
            keep_idx = positive[np.argpartition(row[positive], -top_k)[-top_k:]]
        sparse_graph[idx, keep_idx] = row[keep_idx]
    sparse_graph = np.maximum(sparse_graph, sparse_graph.T)
    np.fill_diagonal(sparse_graph, 0.0)
    return sparse_graph


def resolve_project_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(base_dir, path_value)


def load_and_fuse_graphs(graph_dir, graph_files, graph_weights, normalize, top_k):
    graph_list = []
    stats = []

    for graph_file in graph_files:
        graph_path = os.path.join(graph_dir, graph_file)
        if not os.path.exists(graph_path):
            raise FileNotFoundError('Missing graph file: %s' % graph_path)
        graph = np.load(graph_path).astype(np.float32)
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError('Graph must be square: %s -> %s' % (graph_file, graph.shape))
        graph = normalize_graph(graph, normalize)
        graph = 0.5 * (graph + graph.T)
        np.fill_diagonal(graph, 0.0)
        graph_list.append(graph)
        stats.append(
            {
                'graph_file': graph_file,
                'min': float(graph.min()),
                'max': float(graph.max()),
                'mean': float(graph.mean()),
                'density': float(np.count_nonzero(graph) / graph.size),
            }
        )

    fused = np.zeros_like(graph_list[0])
    weight_sum = float(np.sum(graph_weights))
    for weight, graph in zip(graph_weights, graph_list):
        fused += float(weight) * graph
    fused /= weight_sum
    fused = sparsify_topk(fused, top_k)

    degrees = fused.sum(axis=1)
    if np.any(degrees == 0):
        raise ValueError(
            'Fused graph contains isolated nodes after top_k=%s. Increase top_k or adjust graph_weights.' % top_k
        )

    return fused, stats


def weighted_choice(neighbors, probs, rng):
    cumulative = np.cumsum(probs)
    sample = rng.random() * cumulative[-1]
    choice = int(np.searchsorted(cumulative, sample, side='right'))
    return int(neighbors[choice])


def build_neighbor_cache(graph):
    neighbors = []
    neighbor_sets = []
    weights = []

    for idx in range(graph.shape[0]):
        nbrs = np.where(graph[idx] > 0)[0].astype(np.int64)
        nbr_weights = graph[idx, nbrs].astype(np.float64)
        neighbors.append(nbrs)
        neighbor_sets.append(set(int(item) for item in nbrs))
        weights.append(nbr_weights)

    return neighbors, neighbor_sets, weights


def node2vec_walk(start_node, neighbors, neighbor_sets, weights, walk_length, p, q, rng):
    walk = [int(start_node)]

    while len(walk) < walk_length:
        current = walk[-1]
        current_neighbors = neighbors[current]
        if len(current_neighbors) == 0:
            break

        if len(walk) == 1:
            next_node = weighted_choice(current_neighbors, weights[current], rng)
        else:
            previous = walk[-2]
            bias_weights = []
            for candidate, edge_weight in zip(current_neighbors, weights[current]):
                if candidate == previous:
                    bias = 1.0 / p
                elif candidate in neighbor_sets[previous]:
                    bias = 1.0
                else:
                    bias = 1.0 / q
                bias_weights.append(edge_weight * bias)
            bias_weights = np.asarray(bias_weights, dtype=np.float64)
            next_node = weighted_choice(current_neighbors, bias_weights, rng)

        walk.append(next_node)

    return [str(node) for node in walk]


def generate_walks(graph, walk_length, num_walks, p, q, seed):
    rng = np.random.default_rng(seed)
    neighbors, neighbor_sets, weights = build_neighbor_cache(graph)
    nodes = np.arange(graph.shape[0], dtype=np.int64)
    walks = []

    for _ in range(num_walks):
        order = rng.permutation(nodes)
        for node in order:
            walks.append(
                node2vec_walk(
                    start_node=int(node),
                    neighbors=neighbors,
                    neighbor_sets=neighbor_sets,
                    weights=weights,
                    walk_length=walk_length,
                    p=p,
                    q=q,
                    rng=rng,
                )
            )

    return walks


def train_embeddings(walks, num_nodes, dimensions, window_size, epochs, workers, seed):
    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )
    embedding = np.zeros((num_nodes, dimensions), dtype=np.float32)
    for node_id in range(num_nodes):
        embedding[node_id] = model.wv[str(node_id)]

    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embedding /= norms
    return embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', default=os.path.join('data', 'graph', 'bike'))
    parser.add_argument('--mapping_file', default=os.path.join('data', 'graph', 'bike', 'selected_node_mapping.csv'))
    parser.add_argument('--output', default=os.path.join('data', 'SE', 'se_bike.csv'))
    parser.add_argument('--meta_output', default=None)
    parser.add_argument('--graph_files', default=','.join(DEFAULT_GRAPH_FILES))
    parser.add_argument('--graph_weights', default=None)
    parser.add_argument('--normalize', choices=['max', 'row', 'none'], default='max')
    parser.add_argument('--top_k', type=int, default=12)
    parser.add_argument('--dimensions', type=int, default=144)
    parser.add_argument('--walk_length', type=int, default=40)
    parser.add_argument('--num_walks', type=int, default=40)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.graph_dir = resolve_project_path(PROJECT_ROOT, args.graph_dir)
    args.mapping_file = resolve_project_path(PROJECT_ROOT, args.mapping_file)
    args.output = resolve_project_path(PROJECT_ROOT, args.output)
    if args.meta_output is not None:
        args.meta_output = resolve_project_path(PROJECT_ROOT, args.meta_output)

    graph_files = [item.strip() for item in args.graph_files.split(',') if item.strip()]
    if not graph_files:
        raise ValueError('graph_files cannot be empty')

    graph_weights = parse_graph_weights(args.graph_weights, len(graph_files))
    fused_graph, graph_stats = load_and_fuse_graphs(
        args.graph_dir,
        graph_files,
        graph_weights,
        args.normalize,
        args.top_k,
    )

    walks = generate_walks(
        fused_graph,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        p=args.p,
        q=args.q,
        seed=args.seed,
    )
    embedding = train_embeddings(
        walks,
        num_nodes=fused_graph.shape[0],
        dimensions=args.dimensions,
        window_size=args.window_size,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
    )

    if os.path.exists(args.mapping_file):
        mapping_df = pd.read_csv(args.mapping_file)
        if len(mapping_df) != embedding.shape[0]:
            raise ValueError(
                'Mapping row count does not match embedding rows: %s != %s'
                % (len(mapping_df), embedding.shape[0])
            )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame(embedding).to_csv(args.output, header=False, index=False)

    meta_output = args.meta_output or (args.output + '.meta.json')
    meta = {
        'graph_dir': args.graph_dir,
        'mapping_file': args.mapping_file,
        'output': args.output,
        'graph_files': graph_files,
        'graph_weights': graph_weights.tolist(),
        'normalize': args.normalize,
        'top_k': args.top_k,
        'dimensions': args.dimensions,
        'walk_length': args.walk_length,
        'num_walks': args.num_walks,
        'window_size': args.window_size,
        'epochs': args.epochs,
        'p': args.p,
        'q': args.q,
        'workers': args.workers,
        'seed': args.seed,
        'node_count': int(fused_graph.shape[0]),
        'graph_stats': graph_stats,
        'fused_graph_density': float(np.count_nonzero(fused_graph) / fused_graph.size),
        'fused_graph_min': float(fused_graph.min()),
        'fused_graph_max': float(fused_graph.max()),
        'fused_graph_mean': float(fused_graph.mean()),
    }
    with open(meta_output, 'w', encoding='utf-8') as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    print('Saved node2vec embedding to:', args.output)
    print('Saved metadata to:', meta_output)
    print('Embedding shape:', embedding.shape)
    print('Fused graph density:', meta['fused_graph_density'])
    print('Graph files:', graph_files)


if __name__ == '__main__':
    main()
