import torch
import math

# Probe the e4m3fn max value and saturation behavior
# For OCP MXFP8 E4M3: exp=1111, mant=110 = max = 448.0; exp=1111,mant=111 = NaN
# torch.float8_e4m3fn is IEEE-like: 1e6 saturates to NaN, not 448.

# We need to clamp to 448 before casting
E4M3_MAX = 448.0
E5M2_MAX = 57344.0

x = torch.tensor([400.0, 447.9, 448.0, 448.1, 500.0, -400.0, -448.0, -500.0], dtype=torch.float32)
clamped = x.clamp(-E4M3_MAX, E4M3_MAX)
via_float8 = clamped.to(torch.float8_e4m3fn).to(torch.float32)
print("E4M3 clamped-then-cast:")
for orig, cl, q in zip(x.tolist(), clamped.tolist(), via_float8.tolist()):
    print(f"  raw={orig:+8.2f} -> clamped={cl:+8.2f} -> quant={q:+8.4f}")

# Sanity: can we get 448 out?
t448 = torch.tensor([448.0], dtype=torch.float32)
print(f"\n448.0 through e4m3fn (no clamp): {t448.to(torch.float8_e4m3fn).to(torch.float32).item()}")

# What's the largest value BEFORE 448 that roundtrips correctly?
vals = [400, 416, 432, 440, 448, 480, 512]
print("\nE4M3 roundtrip (clamped to 448):")
for v in vals:
    q = torch.tensor([float(v)], dtype=torch.float32).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32).item()
    print(f"  {v} -> {q}")

# E5M2
print("\nE5M2 test (clamp to 57344):")
vals5 = [50000, 57344, 57345, 65536]
for v in vals5:
    q = torch.tensor([float(v)], dtype=torch.float32).clamp(-E5M2_MAX, E5M2_MAX).to(torch.float8_e5m2).to(torch.float32).item()
    print(f"  {v} -> {q}")

# Check: what does the scale formula give for e4m3 format_max=448?
format_max = 448.0
log2_format_max = math.floor(math.log2(format_max))
print(f"\nE4M3 log2_format_max = floor(log2(448)) = {log2_format_max}")

format_max = 57344.0
log2_format_max = math.floor(math.log2(format_max))
print(f"E5M2 log2_format_max = floor(log2(57344)) = {log2_format_max}")
