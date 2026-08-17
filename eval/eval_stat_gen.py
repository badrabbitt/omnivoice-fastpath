"""Generate the paired evaluation set for one configuration.

Runs inside the omnivoice container. Writes into the shared data volume so the
whisper container can read the same files.

    python eval_stat_gen.py <backend> <name> <num_step>
      backend: "torch" or a path to an .onnx model
"""
import sys, os, time, json, torch, soundfile as sf

torch.set_num_threads(int(os.getenv("EVAL_THREADS", "8")))
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceModelOutput

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))
from eval_sentences import SENTENCES

BACKEND = sys.argv[1]
NAME = sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 32
ROOT = os.getenv("EVAL_ROOT", os.environ.get("OVFP_EVAL_ROOT", "./eval_out"))
OUTDIR = f"{ROOT}/{NAME}"
os.makedirs(OUTDIR, exist_ok=True)

m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32).eval()

if BACKEND != "torch":
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.getenv("EVAL_THREADS", "8"))
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(BACKEND, so, providers=["CPUExecutionProvider"])

    def ort_forward(input_ids=None, audio_mask=None, attention_mask=None, **kw):
        logits = sess.run(None, {
            "input_ids": input_ids.numpy(),
            "audio_mask": audio_mask.numpy(),
            "attention_mask": attention_mask.numpy(),
        })[0]
        return OmniVoiceModelOutput(loss=None, logits=torch.from_numpy(logits))

    m.forward = ort_forward

rows = []
for i, text in enumerate(SENTENCES):
    # Same seed per sentence index across configs keeps the comparison paired.
    torch.manual_seed(1234 + i)
    t0 = time.time()
    audios = m.generate(text=text, language="vi", num_step=STEPS)
    elapsed = time.time() - t0
    path = f"{OUTDIR}/{i:03d}.wav"
    sf.write(path, audios[0], m.sampling_rate)
    dur = len(audios[0]) / m.sampling_rate
    rows.append({"i": i, "text": text, "path": path,
                 "seconds": round(elapsed, 3), "audio": round(dur, 3)})
    if i % 8 == 0:
        print(f"[{NAME}] {i}/{len(SENTENCES)} {elapsed:.1f}s", flush=True)

total = sum(r["seconds"] for r in rows)
audio = sum(r["audio"] for r in rows)
meta = {"name": NAME, "backend": BACKEND, "num_step": STEPS,
        "total_seconds": round(total, 1), "total_audio": round(audio, 1),
        "rtf": round(total / audio, 3), "rows": rows}
with open(f"{OUTDIR}/meta.json", "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"SUMMARY name={NAME} steps={STEPS} total={total:.1f}s audio={audio:.1f}s RTF={total/audio:.2f}x")
