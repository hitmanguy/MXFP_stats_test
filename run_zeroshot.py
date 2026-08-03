import os
import json
import argparse
import torch
import pandas as pd
from typing import Dict, Any

# Ensure lm_eval is available
try:
    from lm_eval.models.huggingface import HFLM
    from lm_eval import simple_evaluate
except ImportError:
    raise ImportError("lm-eval is not installed. Please run: pip install lm-eval")

from frameworks.language import _resolve_modes, _replace_layers

def load_config(path: str) -> Dict:
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_zeroshot_evaluation(cfg: Dict):
    model_name = cfg.get("model_name", "meta-llama/Llama-2-7b-hf")
    quant_modes = cfg.get("quant_modes", ["bf16", "mxfp4", "mxfp4_residual"])
    seeds = cfg.get("seeds", [42, 123, 1337])
    tasks = cfg.get("tasks", ["winogrande", "piqa", "hellaswag", "arc_easy", "boolq"])
    batch_size = cfg.get("batch_size", "auto")
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    block_size = cfg.get("block_size", 32)
    
    print(f"\n{'='*60}")
    print(f"  Zero-Shot Evaluation: {model_name}")
    print(f"  Tasks: {', '.join(tasks)}")
    print(f"  Modes: {', '.join(quant_modes)}")
    print(f"{'='*60}\n")
    
    import collections
    import numpy as np
    import random
    
    # Store all final metrics here for table generation
    all_results = []
    
    for mode in quant_modes:
        print(f"\n--- Testing Quantization Mode: {mode} ---")
        
        mode_metrics = {"format": mode}
        task_scores_all_seeds = collections.defaultdict(list)
        
        for seed in seeds:
            print(f"\n  [Seed {seed}]")
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # 1. Load Base Model inside loop to prevent state pollution
            print(f"  Loading {model_name}...")
            from frameworks.language import LanguageEvalHarness
            harness = LanguageEvalHarness(model_name=model_name, quant_mode="bf16", device=device)
            model, tokenizer = harness._load_model_and_tokenizer()
            
            # 2. Apply Custom Fake-Quantization
            if mode not in ["fp32", "bf16"]:
                print(f"  Applying layer substitution for {mode}...")
                weight_mode, act_mode = _resolve_modes(mode)
                model = _replace_layers(model, weight_mode, act_mode, block_size)
            
            model = model.to(torch.bfloat16).to(device)
            
            # 3. Wrap in lm-eval HFLM
            print(f"  Wrapping model in lm-eval HFLM wrapper...")
            lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
            
            # 4. Run Evaluation
            print(f"  Running simple_evaluate on {len(tasks)} tasks...")
            results = simple_evaluate(
                model=lm_eval_model, 
                tasks=tasks,
                numpy_random_seed=seed,
                torch_random_seed=seed,
                fewshot_random_seed=seed
            )
            
            # 5. Extract Metrics
            for task_name, task_results in results["results"].items():
                if "acc_norm,none" in task_results:
                    score = task_results["acc_norm,none"]
                elif "acc,none" in task_results:
                    score = task_results["acc,none"]
                elif "acc" in task_results:
                    score = task_results["acc"]
                elif "acc_norm" in task_results:
                    score = task_results["acc_norm"]
                else:
                    score = next((v for v in task_results.values() if isinstance(v, float)), 0.0)
                    
                task_scores_all_seeds[task_name].append(score)
                print(f"    [Task: {task_name}] Score: {score:.4f}")
                
            # Clear memory
            del lm_eval_model
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Aggregate scores over seeds
        for task_name, scores in task_scores_all_seeds.items():
            mode_metrics[f"{task_name}_mean"] = float(np.mean(scores))
            mode_metrics[f"{task_name}_std"] = float(np.std(scores))
            
        all_results.append(mode_metrics)
            
    # 6. Save Results
    os.makedirs("results/zeroshot", exist_ok=True)
    safe_name = model_name.replace('/', '_').replace('-', '_')
    
    # Save as JSON
    json_path = f"results/zeroshot/zeroshot_{safe_name}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    # Save as CSV & Markdown
    df = pd.DataFrame(all_results)
    csv_path = f"results/zeroshot/zeroshot_{safe_name}.csv"
    md_path = f"results/zeroshot/zeroshot_{safe_name}.md"
    
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(f"# Zero-Shot Results for {model_name}\n\n")
        f.write(df.to_markdown(index=False))
        
    print(f"\nAll evaluations complete! Results saved to:")
    print(f" - {json_path}")
    print(f" - {csv_path}")
    print(f" - {md_path}")
    print("\nFinal Markdown Table:")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    run_zeroshot_evaluation(cfg)
