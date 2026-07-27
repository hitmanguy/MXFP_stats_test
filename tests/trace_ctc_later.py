import torch
from transformers import Wav2Vec2ForCTC, AutoProcessor
from datasets import load_dataset
from frameworks.speech import _replace_speech_layers

def get_logits(model_name, x, mode):
    model = Wav2Vec2ForCTC.from_pretrained(model_name).cuda()
    model.eval()
    if mode != 'fp32':
        if mode == 'mxfp4': w, a = 'mxfp4', 'mxfp4'
        elif mode == 'act_only': w, a = 'mxfp4', 'mxfp4_residual'
        model = _replace_speech_layers(model, w, a, skip_names=['lm_head'])
    with torch.no_grad():
        return model(x).logits

def main():
    model_name = 'facebook/wav2vec2-base-960h'
    processor = AutoProcessor.from_pretrained(model_name)
    ds = load_dataset('librispeech_asr', 'clean', split='test', streaming=True)
    it = iter(ds)
    
    # Check sample 95 where we know WER gets very bad
    for _ in range(95): sample = next(it)
        
    x = processor(sample['audio']['array'], sampling_rate=16000, return_tensors='pt').input_values.cuda()

    logits_fp32 = get_logits(model_name, x, 'fp32')
    logits_mxfp4 = get_logits(model_name, x, 'mxfp4')
    logits_act = get_logits(model_name, x, 'act_only')

    print("Sample 95 CTC Logits Analysis:")
    for name, logit in [('mxfp4', logits_mxfp4), ('act_only', logits_act)]:
        diff = (logits_fp32 - logit)
        p_fp32 = torch.softmax(logits_fp32, dim=-1)
        p_test = torch.softmax(logit, dim=-1)
        kl = torch.sum(p_fp32 * torch.log(p_fp32 / (p_test + 1e-9) + 1e-9), dim=-1).mean().item()
        pct = (logits_fp32.argmax(dim=-1) != logit.argmax(dim=-1)).float().mean().item() * 100
        print(f"\n[{name}] KL Divergence: {kl:.4f}, Mismatches: {pct:.2f}%")

if __name__ == '__main__': main()
