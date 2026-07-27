import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, AutoProcessor
from datasets import load_dataset

def main():
    model_name = 'facebook/wav2vec2-base-960h'
    processor = AutoProcessor.from_pretrained(model_name)
    
    ds = load_dataset('librispeech_asr', 'clean', split='test', streaming=True)
    sample = next(iter(ds))
    audio = sample['audio']['array']
    inputs = processor(audio, sampling_rate=16000, return_tensors='pt')
    
    x = inputs.input_values.cuda()

    # fp32
    model_fp32 = Wav2Vec2ForCTC.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    model_fp32.eval()
    
    with torch.no_grad():
        out_fp32 = model_fp32(x).logits

    # bf16
    model_bf16 = Wav2Vec2ForCTC.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    model_bf16 = model_bf16.to(torch.bfloat16)
    model_bf16.eval()
    
    with torch.no_grad():
        out_bf16 = model_bf16(x.to(torch.bfloat16)).logits

    diff = (out_fp32.float() - out_bf16.float()).abs()
    print(f"Max diff between fp32 and bf16 logits: {diff.max().item()}")
    print(f"Mean diff between fp32 and bf16 logits: {diff.mean().item()}")
    print(f"Are they exactly identical? {torch.all(out_fp32.float() == out_bf16.float()).item()}")

if __name__ == '__main__':
    main()
