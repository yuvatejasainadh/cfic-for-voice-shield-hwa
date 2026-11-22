import os

files = [
    r"c:\MyProjects\CFIC-VoiceShield-HackVersion\cfic-asr-benchmark\models\NeMo\nemo\collections\asr\parts\preprocessing\segment.py",
    r"c:\MyProjects\CFIC-VoiceShield-HackVersion\cfic-asr-benchmark\models\NeMo\nemo\collections\asr\parts\preprocessing\feature_loader.py"
]

for p in files:
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()
    t_new = text.replace("samples.dtype in np.sctypes['int']", "np.issubdtype(samples.dtype, np.integer)")
    t_new = t_new.replace("samples.dtype in np.sctypes['float']", "np.issubdtype(samples.dtype, np.floating)")
    if t_new != text:
        with open(p, "w", encoding="utf-8") as f:
            f.write(t_new)
        print(f"Successfully patched {os.path.basename(p)}")
    else:
        print(f"No changes needed for {os.path.basename(p)}")
