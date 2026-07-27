from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import time
from sklearn.metrics import roc_auc_score

from core.layers import FakeQuantLinear

def _replace_recsys_layers(
    model: nn.Module,
    weight_mode: str,
    act_mode: str,
    block_size: int,
    quantize_embeddings: bool = False,
) -> nn.Module:
    """
    Recursively replaces nn.Linear with FakeQuantLinear in the DLRM model.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(
                model, name,
                FakeQuantLinear.from_linear(child, weight_mode, act_mode, block_size)
            )
        else:
            _replace_recsys_layers(child, weight_mode, act_mode, block_size, quantize_embeddings)
    return model

class PurePyTorchDLRM(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # 26 categorical features
        self.embeddings = nn.ModuleList([
            nn.EmbeddingBag(10000, 64, mode='sum') for _ in range(26)
        ])
        
        # Bottom MLP for 13 dense features
        self.bottom_mlp = nn.Sequential(
            nn.Linear(13, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        
        # Top MLP: interactions = (26 embeddings + 1 bottom) * (26+1-1)/2 + 64 (bottom out)
        # Actually, standard DLRM concatenates bottom_mlp out + upper triangle of dot products
        # 27 * 26 / 2 = 351 + 64 = 415
        self.top_mlp = nn.Sequential(
            nn.Linear(415, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, dense_features, sparse_features):
        # sparse_features shape: [B, 26]
        # dense_features shape: [B, 13]
        
        # 1. Process sparse features
        emb_outs = []
        for i in range(26):
            idx = sparse_features[:, i].unsqueeze(-1)
            emb_outs.append(self.embeddings[i](idx))
        
        # 2. Process dense features
        bottom_out = self.bottom_mlp(dense_features)
        
        # 3. Feature interaction (dot products)
        all_features = torch.stack([bottom_out] + emb_outs, dim=1)
        interactions = torch.bmm(all_features, all_features.transpose(1, 2))
        
        triu_indices = torch.triu_indices(27, 27, offset=1)
        interacts = interactions[:, triu_indices[0], triu_indices[1]] # [B, 351]
        
        # 4. Top MLP
        top_in = torch.cat([bottom_out, interacts], dim=1)
        return self.top_mlp(top_in)

class DummyCriteoDataset(torch.utils.data.Dataset):
    def __init__(self, size=2048):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        dense = torch.randn(13)
        sparse = torch.randint(0, 10000, (26,))
        label = torch.randint(0, 2, (1,)).float()
        return dense, sparse, label

class RecSysEvalHarness:
    def __init__(
        self,
        quant_mode: str = "fp32",
        n_samples: int = 100_000,
        seed: int = 42,
        block_size: int = 32,
        device: Optional[torch.device] = None,
        quantize_embeddings: bool = False,
    ):
        self.quant_mode = quant_mode
        self.n_samples = n_samples
        self.seed = seed
        self.block_size = block_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantize_embeddings = quantize_embeddings

    def _get_dlrm_model(self):
        print("[INFO] Using PurePyTorchDLRM across all platforms for maximum robustness and consistency.")
        return PurePyTorchDLRM(self.device)

    def _get_dataloader(self):

        print(f"[INFO] Fetching reczoo/Criteo_x1 from Hugging Face Datasets (Zero manual steps required).")
        try:
            from datasets import load_dataset
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError("Please install `datasets` to run the RecSys sweep (pip install datasets).")
        
        # Load real Criteo_x1. We'll load the train split (which is the only split exposed by default in HF for this repo)
        ds = load_dataset('reczoo/Criteo_x1', split='train')
        
        class BatchWrapper:
            def __init__(self, dense, sparse, labels):
                self.dense_features = dense
                self.sparse_features = sparse
                self.labels = labels

        def collate_fn(batch):
            # batch is a list of dictionaries. Map schema: I1..I13, C1..C26, label
            dense_tensors = []
            sparse_tensors = []
            labels = []
            
            for item in batch:
                dense = [item[f'I{i}'] for i in range(1, 14)]
                sparse = [item[f'C{i}'] for i in range(1, 27)]
                # Handle possible NaNs in dense features if the dataset has them
                dense = [0.0 if d is None else d for d in dense]
                
                dense_tensors.append(dense)
                sparse_tensors.append(sparse)
                labels.append([item['label']])
                
            return BatchWrapper(
                dense=torch.tensor(dense_tensors, dtype=torch.float32),
                sparse=torch.tensor(sparse_tensors, dtype=torch.long),
                labels=torch.tensor(labels, dtype=torch.float32)
            )

        # PyTorch DataLoader over HF Dataset
        dataloader = DataLoader(ds, batch_size=2048, collate_fn=collate_fn, shuffle=False)
        return dataloader

    def run(self) -> Dict[str, Any]:
        from core.quantizer import bits_per_value
        from frameworks.language import _resolve_modes

        weight_mode, act_mode = _resolve_modes(self.quant_mode)
        
        print(f"\n{'='*65}")
        print(f"  Model:      PurePyTorch DLRM")
        print(f"  Mode:       {self.quant_mode} (w={weight_mode}, a={act_mode})")
        print(f"  Samples:    {self.n_samples}")
        print(f"  Device:     {self.device}")
        print(f"{'='*65}")

        model = self._get_dlrm_model()
        model.to(self.device)
        model.eval()

        if self.quant_mode not in ("fp32", "bf16"):
            print("  Applying FakeQuantLinear to MLP layers...")
            model = _replace_recsys_layers(
                model,
                weight_mode=weight_mode,
                act_mode=act_mode,
                block_size=self.block_size,
                quantize_embeddings=self.quantize_embeddings
            )
        elif self.quant_mode == "bf16":
            model.to(torch.bfloat16)

        dataloader = self._get_dataloader()

        print("  Running inference...")
        all_labels = []
        all_preds = []
        samples_processed = 0

        with torch.no_grad():
            for batch in dataloader:
                if samples_processed >= self.n_samples:
                    break
                
                dense_features = batch.dense_features.to(self.device)
                sparse_features = batch.sparse_features.to(self.device)
                labels = batch.labels.to(self.device)
                
                if self.quant_mode == "bf16":
                    dense_features = dense_features.to(torch.bfloat16)

                logits = model(dense_features, sparse_features)
                preds = torch.sigmoid(logits.float()).squeeze(-1)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                
                samples_processed += len(labels)
                
                if samples_processed % 20480 == 0:
                    print(f"    [{samples_processed}/{self.n_samples}]")

        all_labels = all_labels[:self.n_samples]
        all_preds = all_preds[:self.n_samples]

        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = 0.5
            print("[WARN] roc_auc_score failed (likely only one class in labels). Returning 0.5")

        eff_bits = bits_per_value(self.quant_mode)
        print(f"\n  > AUC = {auc:.4f}  (eff_bits={eff_bits:.2f})")

        return {
            "auc": auc,
            "n_samples": samples_processed,
            "quant_mode": self.quant_mode,
            "weight_mode": weight_mode,
            "act_mode": act_mode,
            "seed": self.seed,
            "quantize_embeddings": self.quantize_embeddings
        }
