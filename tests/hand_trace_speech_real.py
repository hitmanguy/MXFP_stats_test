import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, AutoProcessor
from datasets import load_dataset
from core.layers import FakeQuantConv1d

def main():
    model_name = 'facebook/wav2vec2-base-960h'
    model = Wav2Vec2ForCTC.from_pretrained(model_name).cuda()
    processor = AutoProcessor.from_pretrained(model_name)
    
    ds = load_dataset('librispeech_asr', 'clean', split='test', streaming=True)
    sample = next(iter(ds))
    audio = sample['audio']['array']
    inputs = processor(audio, sampling_rate=16000, return_tensors='pt')
    input_values = inputs.input_values.cuda()

    # Hook the first Conv1d layer in feature extractor
    target_layer = model.wav2vec2.feature_extractor.conv_layers[0].conv
    
    saved_x = None
    def hook(m, inp, out):
        nonlocal saved_x
        if saved_x is None:
            saved_x = inp[0].detach().clone()
            
    h = target_layer.register_forward_hook(hook)
    with torch.no_grad():
        model(input_values)
    h.remove()
    
    x = saved_x
    
    original = target_layer
    out_ref = original(x)
    
    fq_mxfp4 = FakeQuantConv1d.from_conv1d(original, "mxfp4", "mxfp4")
    out_mxfp4 = fq_mxfp4(x)
    err_mxfp4 = (out_ref - out_mxfp4)
    
    fq_res = FakeQuantConv1d.from_conv1d(original, "mxfp4", "mxfp4_residual")
    out_res = fq_res(x)
    err_res = (out_ref - out_res)
    
    print(f"Conv1d Ref mean abs: {out_ref.abs().mean().item():.4f}")
    print(f"Conv1d mxfp4 mean err:  {err_mxfp4.abs().mean().item():.4f}")
    print(f"Conv1d mxfp4 max err:   {err_mxfp4.abs().max().item():.4f}")
    print(f"Conv1d mxfp4 var err:   {err_mxfp4.var().item():.4f}")
    
    print(f"Conv1d act_only mean:   {err_res.abs().mean().item():.4f}")
    print(f"Conv1d act_only max:    {err_res.abs().max().item():.4f}")
    print(f"Conv1d act_only var:    {err_res.var().item():.4f}")

if __name__ == '__main__':
    main()
