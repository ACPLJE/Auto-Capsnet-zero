import torch
import torch.autograd.profiler as profiler

def estimate_flops_sin(tensor):
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        output = torch.sin(tensor)
    flops = prof.key_averages().flops_total
    return flops

input_tensor = torch.randn(100, 100)
flops = estimate_flops_sin(input_tensor)
print("Estimated FLOPs for sine:", flops)

