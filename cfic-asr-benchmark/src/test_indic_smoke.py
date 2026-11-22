import sys
import time
import os
import torch
import nemo.collections.asr as nemo_asr

sys.stdout.reconfigure(encoding='utf-8')

nemo_path = r"c:\MyProjects\CFIC-VoiceShield-HackVersion\cfic-asr-benchmark\models\indicconformer_stt_te_hybrid_rnnt_large.nemo"
test_wav = r"c:\MyProjects\CFIC-VoiceShield-HackVersion\cfic-asr-benchmark\data\audio\telugu\telugu_00000.wav"

print(f"Loading IndicConformer Telugu model from {nemo_path}...")
t0 = time.time()
try:
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(restore_path=nemo_path, map_location=torch.device('cpu'))
    init_time = time.time() - t0
    print(f"Model successfully loaded in {init_time:.2f}s!")
    model.eval()
    
    print(f"Current decoder: {getattr(model, 'cur_decoder', None)}")
    print(f"Transcribing {test_wav} with language_id='te'...")
    t1 = time.time()
    out = model.transcribe([test_wav], language_id='te')
    inf_time = time.time() - t1
    print(f"Inference complete in {inf_time:.2f}s!")
    print(f"Output type: {type(out)}, content: {out}")
except Exception as e:
    import traceback
    traceback.print_exc()
