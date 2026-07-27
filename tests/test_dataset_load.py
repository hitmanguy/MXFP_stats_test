"""Quick test: load WikiText-2 with SSL cert fix applied."""
import ssl
import os
import sys

# Fix Windows SSL cert store corruption - use certifi bundle instead
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    print(f"SSL_CERT_FILE set to: {certifi.where()}")
except ImportError:
    print("certifi not available, trying without cert fix")

from datasets import load_dataset

print("Loading WikiText-2...")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
print(f"Loaded: {len(ds)} rows")
print(f"Sample: {repr(ds['text'][1][:80])}")
print("SUCCESS")
