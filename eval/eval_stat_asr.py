"""Transcribe every generated clip with faster-whisper large-v3.

Runs inside the whisper container (faster_whisper is already installed there).
Writes <root>/<config>/asr.json for the stats step.
"""
import json, os, sys, time
from pathlib import Path

ROOT = os.getenv("EVAL_ROOT", os.environ.get("OVFP_EVAL_ROOT", "./eval_out"))
MODEL = os.getenv("EVAL_ASR_MODEL", "large-v3")
CONFIGS = sys.argv[1:] or sorted(
    d.name for d in Path(ROOT).iterdir() if d.is_dir()
)

from faster_whisper import WhisperModel

t0 = time.time()
model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=8)
print(f"loaded {MODEL} in {time.time()-t0:.1f}s", flush=True)

for cfg in CONFIGS:
    d = Path(ROOT) / cfg
    wavs = sorted(d.glob("*.wav"))
    if not wavs:
        print(f"skip {cfg}: no wavs", flush=True)
        continue
    out = {}
    t0 = time.time()
    for w in wavs:
        # beam_size=5 and no VAD: we want a faithful read of what was synthesized.
        segments, _ = model.transcribe(str(w), language="vi", beam_size=5, vad_filter=False)
        out[w.stem] = " ".join(s.text.strip() for s in segments).strip()
    (d / "asr.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ASR_DONE {cfg} n={len(out)} in {time.time()-t0:.0f}s", flush=True)
