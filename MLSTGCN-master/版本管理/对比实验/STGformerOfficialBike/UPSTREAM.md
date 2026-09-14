# Upstream provenance

- Project: STGformer
- Repository: https://github.com/Dreamzz5/STGformer
- Paper: https://arxiv.org/abs/2410.00385
- Pinned commit: `7c141029328a91cfeb78c1d8a0bfaa26997d30a0`
- Imported on: 2026-09-14
- Upstream file: `model/STGformer.py`
- Upstream license: MIT, copied to `LICENSE`
- Canonical source SHA-256: `7afde2e696ebd36b58564946aa3c0a69916f57853a556b6964a5047c0cc78a44`
- Canonical license SHA-256: `eb5f9581e23954ff1e10194c31b66c486c361f218e9241bfb9c70af5d8583de0`

The canonical hashes normalize CRLF to LF and ignore trailing blank lines.

`upstream/STGformer.py` is an unmodified source snapshot from the pinned
commit. It is retained for provenance and code review, not imported by the
local training entry point because the upstream module has a hard dependency
on `timm`, which is absent from the project environment.

The executable adaptation is in `models/stgformer_official_adapter.py`. Its
deliberate differences from the snapshot are limited to:

1. an in-file MLP equivalent to the `timm` MLP used upstream;
2. explicit `tod` and `dow` tensors derived from the normalized local hourly
   features, instead of overloading fixed input channels 1 and 2;
3. PyTorch 1.9-compatible calls (`torch.cat` and `reshape`);
4. input/shape validation and a public adaptive-graph diagnostic method;
5. the local Top-150, history-168, horizon-3, two-channel output contract.
6. out-of-place residual accumulation equivalent to the upstream recurrence,
   avoiding mutation of the caller's input tensor during autograd.

The upstream model constructs one learned adaptive graph from its adaptive
embedding. The external `dist.npy` graph is therefore used only to audit node
count/order for B0; it is not injected into the forward pass. Adding external
or multiple relation graphs belongs to B1 and later experiments.
