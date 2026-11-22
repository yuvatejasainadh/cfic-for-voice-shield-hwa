import sys
import torch
import nemo.collections.asr as nemo_asr

nemo_path = r"c:\MyProjects\CFIC-VoiceShield-HackVersion\cfic-asr-benchmark\models\indicconformer_stt_te_hybrid_rnnt_large.nemo"
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(restore_path=nemo_path, map_location=torch.device('cpu'))

print("CTC decoder language_masks keys:", list(model.ctc_decoder.language_masks.keys()) if hasattr(model.ctc_decoder, 'language_masks') else "None")
print("Joint language_masks keys:", list(model.joint.language_masks.keys()) if hasattr(model.joint, 'language_masks') else "None")
if hasattr(model, 'tokenizer'):
    print("Tokenizer languages:", getattr(model.tokenizer, 'languages', None))
    print("Tokenizer vocab:", type(model.tokenizer))
