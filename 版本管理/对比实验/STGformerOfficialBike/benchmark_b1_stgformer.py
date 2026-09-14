"""Compare B0/B1 checkpoint latency and allocated GPU memory on one device."""
import argparse
import time
import numpy as np
import torch
from models import build_model_from_checkpoint
from models.multi_relation_graph_propagate import build_b1_from_checkpoint
from protocol import resolve_device, save_json, environment_info
from train_stgformer_official_bike import DEFAULT_RESULT_ROOT


def measure(checkpoint, builder, device, batch, repetitions):
    model, _ = builder(checkpoint, device)
    model.eval()
    c = checkpoint['model_config']
    x = torch.zeros(batch, c['in_steps'], c['num_nodes'], c['input_dim'], device=device)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(repetitions):
            sync()
            start = time.perf_counter()
            model(x)
            sync()
            times.append(1000*(time.perf_counter()-start))
    result = dict(parameters=sum(p.numel() for p in model.parameters()),
                  median_latency_ms=float(np.median(times)),
                  peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None)
    del model, x
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--repetitions', type=int, default=30)
    args = p.parse_args()
    if args.repetitions < 1 or args.batch_size < 1:
        raise ValueError('repetitions and batch_size must be positive')
    root = DEFAULT_RESULT_ROOT
    b0 = torch.load(root / ('b0_top150_hist168_pred3_seed%d_bs16' % args.seed) /
                    'best_stgformer_official_b0.pt', map_location='cpu')
    b1_dir = root / ('b1_static_multirelation_seed%d_bs16' % args.seed)
    b1 = torch.load(b1_dir / 'best_stgformer_b1.pt', map_location='cpu')
    for key in ['in_steps', 'out_steps', 'num_nodes', 'input_dim', 'output_dim']:
        if b0['model_config'][key] != b1['model_config'][key]:
            raise ValueError('Benchmark shape mismatch: '+key)
    device = resolve_device(args.device)
    old = measure(b0, build_model_from_checkpoint, device, args.batch_size, args.repetitions)
    new = measure(b1, build_b1_from_checkpoint, device, args.batch_size, args.repetitions)
    result = dict(b0=old, b1=new, args=vars(args), environment=environment_info(),
                  latency_increase_pct=100*(new['median_latency_ms']/old['median_latency_ms']-1),
                  memory_increase_pct=100*(new['peak_allocated_bytes']/old['peak_allocated_bytes']-1)
                  if old['peak_allocated_bytes'] else None,
                  input='normalized zeros; 5 warmups; sequential isolated models')
    save_json(b1_dir / 'efficiency.json', result)
    print(result)


if __name__ == '__main__':
    main()
