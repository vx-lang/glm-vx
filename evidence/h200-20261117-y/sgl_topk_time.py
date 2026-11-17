"""SGLang's indexer top-k (sgl_kernel.fast_topk_v2, the decode backend 'sgl-kernel') at batch 1 on
the H200: CUDA events after an L2 flush, median of 20, the same tokens as validation/topk_warp_ab."""
import torch, sgl_kernel
from sgl_kernel import fast_topk_v2
flush = torch.empty(256 << 20, dtype=torch.uint8, device='cuda')
for tokens in [2049, 3001, 4096, 8192, 16384, 32768, 65536, 131072, 200000]:
    g = torch.Generator(device='cuda').manual_seed(1)
    score = torch.randn((1, tokens), device='cuda', generator=g, dtype=torch.float32)
    lengths = torch.tensor([tokens], device='cuda', dtype=torch.int32)
    out = fast_topk_v2(score, lengths, 2048)
    want = set(torch.topk(score[0], 2048).indices.tolist())
    ok = set(out[0].tolist()) == want
    times = []
    for _ in range(20):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fast_topk_v2(score, lengths, 2048); b.record(); torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    times.sort()
    print(f'| {tokens} | {times[10]:.1f} | {"same set" if ok else "DIFFERENT SET"} |', flush=True)
