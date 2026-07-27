import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, AutoProcessor
from datasets import load_dataset
from frameworks.speech import _replace_speech_layers

def get_logits(model_name, x, mode):
    model = Wav2Vec2ForCTC.from_pretrained(model_name).cuda()
    model.eval()
    if mode == 'fp32':
        pass
    else:
        if mode == 'mxfp4':
            w, a = 'mxfp4', 'mxfp4'
        elif mode == 'act_only':
            w, a = 'mxfp4', 'mxfp4_residual'
        elif mode == 'weight_only':
            w, a = 'mxfp4_residual', 'mxfp4'
        elif mode == 'full':
            w, a = 'mxfp4_residual', 'mxfp4_residual'
            
        model = _replace_speech_layers(model, w, a, skip_names=['lm_head'])
        
    with torch.no_grad():
        out = model(x).logits
    return out

def main():
    model_name = 'facebook/wav2vec2-base-960h'
    processor = AutoProcessor.from_pretrained(model_name)
    
    ds = load_dataset('librispeech_asr', 'clean', split='test', streaming=True)
    it = iter(ds)
    # Skip to 60th sample where WER is bad
    for _ in range(60):
        sample = next(it)
        
    audio = sample['audio']['array']
    inputs = processor(audio, sampling_rate=16000, return_tensors='pt')
    x = inputs.input_values.cuda()

    logits_fp32 = get_logits(model_name, x, 'fp32')
    logits_mxfp4 = get_logits(model_name, x, 'mxfp4')
    logits_act = get_logits(model_name, x, 'act_only')
    logits_weight = get_logits(model_name, x, 'weight_only')
    logits_full = get_logits(model_name, x, 'full')

    print("CTC Logits Analysis:")
    for name, logit in [('mxfp4', logits_mxfp4), ('act_only', logits_act), ('weight_only', logits_weight), ('full', logits_full)]:
        diff = (logits_fp32 - logit)
        print(f"\n[{name}]")
        print(f"Mean Abs Error vs FP32: {diff.abs().mean().item():.4f}")
        print(f"Max Abs Error vs FP32:  {diff.abs().max().item():.4f}")
        
        # Calculate divergence in probability distribution (softmax)
        p_fp32 = torch.softmax(logits_fp32, dim=-1)
        p_test = torch.softmax(logit, dim=-1)
        kl = torch.sum(p_fp32 * torch.log(p_fp32 / (p_test + 1e-9) + 1e-9), dim=-1).mean().item()
        print(f"KL Divergence vs FP32:  {kl:.4f}")
        
        # How often does the greedy prediction change?
        pred_fp32 = logits_fp32.argmax(dim=-1)
        pred_test = logit.argmax(dim=-1)
        pct_change = (pred_fp32 != pred_test).float().mean().item() * 100
        print(f"Greedy token mismatches vs FP32: {pct_change:.2f}%")

if __name__ == '__main__':
    main()
