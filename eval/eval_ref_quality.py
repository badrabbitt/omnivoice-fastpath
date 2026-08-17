"""Generate the paired eval set in voice-cloning mode for a given reference.

Used to test whether a shorter reference clip costs quality. WER only measures
intelligibility, not voice similarity — listen to the output too.

    python eval_ref_quality.py <name> <ref.wav> <ref_text> [num_step]
"""
import os, sys, time, json

import torch
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))
from eval_sentences import SENTENCES

NAME, REF, REF_TEXT = sys.argv[1], sys.argv[2], sys.argv[3]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 32
DEVICE = os.getenv("EVAL_DEVICE", "mps")
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32
ROOT = os.getenv("EVAL_ROOT_HOST", os.environ.get("OVFP_EVAL_ROOT", "./eval_out"))
OUTDIR = f"{ROOT}/{NAME}"
os.makedirs(OUTDIR, exist_ok=True)

from omnivoice.models.omnivoice import OmniVoice
from omnivoice_fastpath import apply_trimmed_cfg, apply_cfg_interval

m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=DEVICE, dtype=DTYPE).eval()
apply_trimmed_cfg(m)
CFG_KEEP = float(os.getenv("EVAL_CFG_KEEP", "1.0"))
if CFG_KEEP < 1.0:
    apply_cfg_interval(m, keep=CFG_KEEP, where=os.getenv("EVAL_CFG_WHERE", "first"))
prompt = m.create_voice_clone_prompt(ref_audio=REF, ref_text=REF_TEXT)

rows = []
t_all = time.time()
for i, text in enumerate(SENTENCES):
    torch.manual_seed(1234 + i)
    t0 = time.time()
    audios = m.generate(text=text, language="vi", voice_clone_prompt=prompt, num_step=STEPS)
    elapsed = time.time() - t0
    path = f"{OUTDIR}/{i:03d}.wav"
    sf.write(path, audios[0], m.sampling_rate)
    rows.append({"i": i, "text": text, "path": path,
                 "seconds": round(elapsed, 3),
                 "audio": round(len(audios[0]) / m.sampling_rate, 3)})

total = sum(r["seconds"] for r in rows)
audio = sum(r["audio"] for r in rows)
meta = {"name": NAME, "backend": f"{DEVICE}/{DTYPE}".replace("torch.", ""),
        "ref": REF, "ref_text": REF_TEXT, "num_step": STEPS, "cfg_keep": CFG_KEEP,
        "total_seconds": round(total, 1), "total_audio": round(audio, 1),
        "rtf": round(total / audio, 3), "rows": rows}
with open(f"{OUTDIR}/meta.json", "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"SUMMARY name={NAME} steps={STEPS} ref={os.path.basename(REF)} "
      f"total={total:.1f}s audio={audio:.1f}s RTF={total/audio:.2f}x")
