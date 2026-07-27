import os
import sys

# Ensure SSL works
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frameworks.language import LanguageEvalHarness

if __name__ == "__main__":
    h = LanguageEvalHarness("gpt2", quant_mode="mxfp4_adaptive_18", seed=42, n_chunks=5)
    res = h.run()
    print("mxfp4_adaptive_18 PPL:", res["ppl"])
    
    h2 = LanguageEvalHarness("gpt2", quant_mode="mxfp4_adaptive_15", seed=42, n_chunks=50)
    res2 = h2.run()
    print("mxfp4_adaptive_15 PPL:", res2["ppl"])
