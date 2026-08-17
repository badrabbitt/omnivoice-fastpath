"""How much does reference-audio length cost?

Every diffusion step runs attention over prefix+target, so a longer reference
inflates the sequence for all 32 steps. Measures RTF and the actual sequence
length the model ends up running.

    python bench_reflen.py --device mps --dtype fp16 --steps 32
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sentences import SENTENCES

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}

# Natural clips of different lengths, each paired with its exact transcript.
# Trimming one recording and guessing where the transcript should be cut skews
# the model's speaking-rate estimate, which contaminates the measurement.
EVAL_DIR = os.environ.get("OVFP_REF_DIR", "./refs")
REF_IDS = [12, 11, 9, 0]  # ~3.1s, 4.0s, 5.2s, 6.2s
REFS = [(f"{EVAL_DIR}/{i:03d}.wav", SENTENCES[i]) for i in REF_IDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from omnivoice.models.omnivoice import OmniVoice

    m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=args.device,
                                  dtype=DTYPES[args.dtype]).eval()

    seqlens = []
    orig = m.forward

    def spy(input_ids=None, **kw):
        seqlens.append(int(input_ids.shape[-1]))
        return orig(input_ids=input_ids, **kw)

    m.forward = spy

    texts = SENTENCES[: args.n]
    rows = []
    for path, ref_text in REFS:
        prompt = m.create_voice_clone_prompt(ref_audio=path, ref_text=ref_text)
        ref_tokens = int(prompt.ref_audio_tokens.shape[-1])
        # Warm up once so the first timed item is not paying lazy-init cost.
        m.generate(text=texts[0], language="vi", voice_clone_prompt=prompt, num_step=4)
        seqlens.clear()
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        audio_s = 0.0
        for t in texts:
            outs = m.generate(text=t, language="vi", voice_clone_prompt=prompt,
                              num_step=args.steps)
            audio_s += len(outs[0]) / m.sampling_rate
        if args.device == "cuda":
            torch.cuda.synchronize()
        total = time.time() - t0
        row = {"ref": os.path.basename(path), "ref_tokens": ref_tokens,
               "mean_seqlen": round(sum(seqlens) / len(seqlens), 1),
               "total_seconds": round(total, 2), "audio_seconds": round(audio_s, 2),
               "rtf": round(total / audio_s, 3)}
        rows.append(row)
        print("REFLEN " + json.dumps(row), flush=True)

    # Wall time for the same input texts is the honest comparison: RTF also
    # moves when the reference changes the model's pacing estimate.
    base = rows[-1]
    print()
    for r in rows:
        print(f"  {r['ref']:<14} ref_tokens={r['ref_tokens']:>4} seq={r['mean_seqlen']:>6}  "
              f"tổng={r['total_seconds']:>6.2f}s  RTF={r['rtf']:<6} "
              f"nhanh hơn {base['total_seconds'] / r['total_seconds']:.2f}x")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
