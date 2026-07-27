import torch
from core.quantizer import fake_quant_mxfp4, fake_quant_mxfp4_residual

torch.manual_seed(42)

def act_only_trace(x):
    # This exactly mimics _quantise_activation(x, 'mxfp4_residual_act_only') 
    # BUT wait, w=mxfp4, a=mxfp4_residual means it calls fake_quant_mxfp4_residual!
    print(f"Original shape: {x.shape}")
    x_flat = x.reshape(-1, 32).float()
    print(f"Blocked shape: {x_flat.shape}")

    # Primary pass
    amax = torch.amax(torch.abs(x_flat), dim=-1, keepdim=True)
    
    # Let's inspect block 0
    b0 = x_flat[0]
    b0_amax = amax[0].item()
    print(f"\nBlock 0 original: {b0.tolist()[:5]}...")
    print(f"Block 0 amax: {b0_amax}")

    # Quantize
    x_q1 = fake_quant_mxfp4(x, 32)
    res = x - x_q1
    x_q2 = fake_quant_mxfp4_residual(x, 32)
    
    # Block 0 residual
    print(f"Block 0 q1: {x_q1.reshape(-1,32)[0].tolist()[:5]}...")
    print(f"Block 0 res: {res.reshape(-1,32)[0].tolist()[:5]}...")
    print(f"Block 0 q2: {x_q2.reshape(-1,32)[0].tolist()[:5]}...")

    err1 = (x - x_q1).abs().mean().item()
    err2 = (x - x_q2).abs().mean().item()
    print(f"\nMean error q1: {err1:.6f}")
    print(f"Mean error q2: {err2:.6f}")

x = torch.randn(1, 1, 512)
act_only_trace(x)
