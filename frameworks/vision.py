"""
frameworks/vision.py
====================
ImageNet Top-1/Top-5 evaluation harness for vision models.
Part 5 — only active after Part 3 acceptance gate passes.

Target models: resnet18, resnet50, deit_small_patch16_224
Dataset: ImageNet ILSVRC12 (imagenet-1k from HuggingFace datasets)
Metrics: Top-1 Accuracy %, Top-5 Accuracy %

Layer Interceptors: Replaces torch.nn.Conv2d and torch.nn.Linear.

NOTE: This module is a stub for Part 5. Activate only after Part 3 gate passes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from core.layers import FakeQuantLinear


class FakeQuantConv2d(nn.Module):
    """
    Fake-quant wrapper for nn.Conv2d.
    Quantises weights (pre-quantised once) and activations (per forward).
    Blocks are formed by flattening weight kernel [out, in, kH, kW] along
    the in*kH*kW dimension (input channel locality).
    """

    def __init__(
        self,
        original: nn.Conv2d,
        weight_mode: str = "mxfp4",
        act_mode: str = "mxfp4",
        block_size: int = 32,
    ):
        super().__init__()
        import torch.nn.functional as F
        self._F = F
        self.weight = nn.Parameter(original.weight.data.clone())
        self.bias = nn.Parameter(original.bias.data.clone()) if original.bias is not None else None
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups
        self.weight_mode = weight_mode
        self.act_mode = act_mode
        self.block_size = block_size

        # Pre-quantise weights once. Flatten spatial dims for blocking.
        if weight_mode not in ("fp32", "bf16", "none"):
            from core.layers import _quantise_weight
            w_flat = self.weight.data.flatten(1)   # [out, in*kH*kW]
            w_q, _ = _quantise_weight(w_flat, weight_mode, block_size)
            self.weight.data = w_q.reshape(original.weight.shape).to(self.weight.data.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from core.layers import _quantise_activation
        dev = x.device
        # ── Weight ──────────────────────────────────────────────────────────
        w = self.weight.to(device=dev, dtype=x.dtype)
            
        x_q, _ = _quantise_activation(x.float(), self.act_mode, self.block_size)
        x_q = x_q.to(x.dtype)
        bias = self.bias.to(device=dev, dtype=x.dtype) if self.bias is not None else None
        return self._F.conv2d(x_q, w, bias, self.stride, self.padding, self.dilation, self.groups)


def replace_vision_layers(model: nn.Module, weight_mode: str, act_mode: str, block_size: int = 32) -> nn.Module:
    """Walk model and replace Conv2d/Linear with FakeQuant equivalents."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Conv2d):
            setattr(model, name, FakeQuantConv2d(module, weight_mode, act_mode, block_size))
        elif isinstance(module, nn.Linear):
            setattr(model, name, FakeQuantLinear.from_linear(module, weight_mode, act_mode, block_size))
        else:
            replace_vision_layers(module, weight_mode, act_mode, block_size)
    return model


class VisionEvalHarness:
    """
    ImageNet Top-1/Top-5 evaluation harness.
    Requires: pip install timm (for DeiT)
    Dataset: HuggingFace imagenet-1k (requires auth)
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        quant_mode: str = "fp32",
        n_batches: int = 100,
        batch_size: int = 32,
        seed: int = 42,
        block_size: int = 32,
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.quant_mode = quant_mode
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.seed = seed
        self.block_size = block_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> Dict[str, Any]:
        from frameworks.language import _resolve_modes
        weight_mode, act_mode = _resolve_modes(self.quant_mode)
        
        # 1. Load Model
        import torchvision
        from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
        
        print(f"\n  Loading {self.model_name}...")
        if self.model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            transforms = weights.transforms()
        elif self.model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
            transforms = weights.transforms()
        elif self.model_name.startswith("deit_"):
            try:
                import timm
            except ImportError:
                raise ImportError("Please install 'timm' to run DeiT models (pip install timm).")
            # Create DeiT model via timm
            model = timm.create_model(self.model_name, pretrained=True)
            # Create standard ImageNet transforms for timm models
            from torchvision import transforms as T
            transforms = T.Compose([
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
        model.eval()
        
        # 2. Apply Quantization
        if self.quant_mode != "fp32":
            print(f"  Applying layer substitution (w={weight_mode}, a={act_mode})...")
            model = replace_vision_layers(model, weight_mode, act_mode, self.block_size)
            print("  Layer substitution complete.")
            
        model = model.to(self.device)
        
        # 3. Load Dataset
        import datasets
        print("  Loading ImageNet-1k validation split (streaming)...")
        try:
            ds = datasets.load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, trust_remote_code=True, token=True)
        except Exception as e:
            print("\n  !! ERROR: Failed to load ILSVRC/imagenet-1k.")
            print("  Make sure you have accepted the terms on HuggingFace and set HF_TOKEN.")
            raise e
            
        ds = ds.shuffle(seed=self.seed, buffer_size=1000)
        ds_iter = iter(ds)
        
        # 4. Evaluate
        correct_1 = 0
        correct_5 = 0
        total = 0
        
        print(f"  Evaluating {self.n_batches} batches of size {self.batch_size}...")
        
        with torch.no_grad():
            for b in range(self.n_batches):
                images = []
                targets = []
                # Fetch batch manually from stream
                try:
                    for _ in range(self.batch_size):
                        example = next(ds_iter)
                        # Ensure image is RGB
                        img = example["image"].convert("RGB")
                        images.append(transforms(img))
                        targets.append(example["label"])
                except StopIteration:
                    break
                    
                if not images:
                    break
                    
                x = torch.stack(images).to(self.device)
                y = torch.tensor(targets, dtype=torch.long, device=self.device)
                
                logits = model(x)
                
                # Top-1 and Top-5
                _, pred = logits.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(y.view(1, -1).expand_as(pred))
                
                correct_1 += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
                correct_5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
                total += y.size(0)
                
                if (b + 1) % max(1, self.n_batches // 10) == 0:
                    print(f"    Batch {b+1}/{self.n_batches} | Acc@1: {correct_1/total*100:.2f}% | Acc@5: {correct_5/total*100:.2f}%")

        acc1 = correct_1 / total * 100.0 if total > 0 else 0.0
        acc5 = correct_5 / total * 100.0 if total > 0 else 0.0
        
        print(f"\n  ✓ Acc@1 = {acc1:.2f}% | Acc@5 = {acc5:.2f}%")
        
        from core.quantizer import bits_per_value
        return {
            "acc1": acc1,
            "acc5": acc5,
            "total_samples": total,
            "eff_bits": bits_per_value(self.quant_mode)
        }
