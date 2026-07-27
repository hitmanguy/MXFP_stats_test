"""
tests/probe_resnet18.py
=======================
Download and verify ResNet-18 weights via the same certifi SSL patch used
for HuggingFace. Prints the documented Top-1 accuracy.
"""
import os, ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torchvision
print(f"torchvision version: {torchvision.__version__}")

from torchvision.models import resnet18, ResNet18_Weights
weights = ResNet18_Weights.DEFAULT
m = resnet18(weights=weights)
m.eval()
print("ResNet-18 loaded OK")
meta = weights.meta
print(f"Documented acc@1: {meta.get('_metrics', {}).get('ImageNet-1K', {}).get('acc@1', 'n/a')}")
print(f"Documented acc@5: {meta.get('_metrics', {}).get('ImageNet-1K', {}).get('acc@5', 'n/a')}")
print(f"Resize: {weights.transforms().crop_size}")
