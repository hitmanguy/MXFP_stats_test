import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, AutoProcessor
from datasets import load_dataset
from frameworks.speech import _replace_speech_layers

def main():
    model_name = 'facebook/wav2vec2-base-960h'
    processor = AutoProcessor.from_pretrained(model_name)
    
    ds = load_dataset('librispeech_asr', 'clean', split='test', streaming=True)
    it = iter(ds)
    # Skip to 60th sample where WER is known to be bad for act_only
    for _ in range(60):
        sample = next(it)
        
    audio = sample['audio']['array']
    inputs = processor(audio, sampling_rate=16000, return_tensors='pt')
    x = inputs.input_values.cuda()

    # Model 1: fp32
    model_ref = Wav2Vec2ForCTC.from_pretrained(model_name).cuda()
    model_ref.eval()
    
    # Model 2: act_only
    model_act = Wav2Vec2ForCTC.from_pretrained(model_name).cuda()
    model_act.eval()
    model_act = _replace_speech_layers(model_act, 'mxfp4', 'mxfp4_residual', skip_names=['lm_head'])
    
    # Model 3: mxfp4
    model_mxfp4 = Wav2Vec2ForCTC.from_pretrained(model_name).cuda()
    model_mxfp4.eval()
    model_mxfp4 = _replace_speech_layers(model_mxfp4, 'mxfp4', 'mxfp4', skip_names=['lm_head'])

    # We will hook the outputs of every layer
    errs_act = {}
    errs_mxfp4 = {}
    
    ref_outputs = {}
    def get_ref_hook(name):
        def hook(m, inp, out):
            ref_outputs[name] = out[0].detach() if isinstance(out, tuple) else out.detach()
        return hook

    for name, module in model_ref.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.register_forward_hook(get_ref_hook(name))
            
    with torch.no_grad():
        model_ref(x)
        
    def get_test_hook(name, err_dict):
        def hook(m, inp, out):
            val = out[0].detach() if isinstance(out, tuple) else out.detach()
            ref = ref_outputs[name]
            err = (val - ref).abs().mean().item()
            err_dict[name] = err
        return hook

    for name, module in model_act.named_modules():
        if getattr(module, 'weight_mode', None) is not None:
            module.register_forward_hook(get_test_hook(name, errs_act))
            
    for name, module in model_mxfp4.named_modules():
        if getattr(module, 'weight_mode', None) is not None:
            module.register_forward_hook(get_test_hook(name, errs_mxfp4))
            
    with torch.no_grad():
        model_act(x)
        model_mxfp4(x)
        
    print(f"{'Layer Name':<60} {'mxfp4 err':<15} {'act_only err':<15}")
    for name in errs_act:
        e1 = errs_mxfp4[name]
        e2 = errs_act[name]
        marker = "<--- WORSE!" if e2 > e1 * 1.5 else ""
        print(f"{name:<60} {e1:<15.4f} {e2:<15.4f} {marker}")

if __name__ == '__main__':
    main()
