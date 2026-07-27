import torch
import math

# Pure Python reference for MXFP4 quantization
def get_mxfp4_levels():
    return [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def quantize_block_pure_python(block: list[float]):
    # 1. compute amax
    abs_block = [abs(x) for x in block]
    amax = max(abs_block)
    
    # 2. compute scale
    if amax == 0.0:
        scale = 1.0
    else:
        # floor(log2(amax)) - floor(log2(6))
        scale_exp = math.floor(math.log2(amax)) - 2
        scale = 2.0 ** scale_exp
        
    # 3. quantize each element
    levels = get_mxfp4_levels()
    quant_vals = []
    
    for x in block:
        x_scaled = abs(x) / scale
        x_clamped = min(x_scaled, 6.0)
        
        # find nearest level
        min_dist = float('inf')
        nearest_idx = -1
        
        for i, lvl in enumerate(levels):
            dist = abs(x_clamped - lvl)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
            elif dist == min_dist:
                # tie breaker: round to nearest even (even index in levels)
                # the previous nearest_idx was i-1, current is i
                if (i - 1) % 2 == 0:
                    nearest_idx = i - 1
                else:
                    nearest_idx = i
                    
        quant_vals.append(levels[nearest_idx] * (1.0 if x >= 0 else -1.0) * scale)
        
    return quant_vals, scale

# Now compare with PyTorch implementation
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.quantizer import fake_quant_mxfp4, _compute_e8m0_scale

def test_hand_trace():
    # Use a realistic block, including values near midpoints
    # For example, scale = 1.0, midpoints 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    # Let's generate a synthetic block that hits exact midpoints, slightly above/below, and bounds
    block = [
        0.0, 0.25, -0.25, 0.251, 0.249, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
        6.0, 7.0, 8.0, 0.1, -1.25, -2.5, -3.5, -5.0, 1.0, 2.0, 3.0, 4.0
    ]
    
    # Pad to 32 elements
    while len(block) < 32:
        block.append(0.0)
        
    print("Testing Synthetic Block")
    py_quant, py_scale = quantize_block_pure_python(block)
    
    t_block = torch.tensor(block, dtype=torch.float32)
    pt_quant = fake_quant_mxfp4(t_block.unsqueeze(0)).squeeze(0).tolist()
    
    amax = t_block.abs().max().unsqueeze(0)
    pt_scale = _compute_e8m0_scale(amax).item()
    
    print(f"Scale: Py={py_scale}, PT={pt_scale}")
    if py_scale != pt_scale:
        print("SCALE MISMATCH!")
        
    mismatches = 0
    for i in range(32):
        if py_quant[i] != pt_quant[i]:
            print(f"Index {i}, val={block[i]}: Py={py_quant[i]}, PT={pt_quant[i]}")
            mismatches += 1
            
    if mismatches == 0:
        print("PERFECT MATCH for synthetic block.")
    else:
        print(f"{mismatches} mismatches found!")

    # Now let's try 1000 random blocks to be sure
    torch.manual_seed(42)
    import random
    random.seed(42)
    
    total_mismatches = 0
    for _ in range(100):
        # Generate random values between -10 and 10
        block = [random.uniform(-10, 10) for _ in range(32)]
        py_quant, py_scale = quantize_block_pure_python(block)
        
        t_block = torch.tensor(block, dtype=torch.float32)
        pt_quant = fake_quant_mxfp4(t_block.unsqueeze(0)).squeeze(0).tolist()
        
        for i in range(32):
            # Check within float precision
            if abs(py_quant[i] - pt_quant[i]) > 1e-6:
                print(f"Random block mismatch! val={block[i]}, scale={py_scale}: Py={py_quant[i]}, PT={pt_quant[i]}")
                total_mismatches += 1
                break
                
    print(f"Random testing complete. {total_mismatches} mismatches.")

if __name__ == "__main__":
    test_hand_trace()
