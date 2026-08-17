"""Measure the saving from running CFG on only part of the diffusion steps."""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sentences import SENTENCES

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--keeps", default="1.0,0.75,0.5,0.25")
    ap.add_argument("--where", default="first")
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from omnivoice.models.omnivoice import OmniVoice
    from omnivoice_fastpath import apply_trimmed_cfg, apply_cfg_interval

    texts = sorted(SENTENCES, key=len)[: args.n]
    rows = []

    for keep in [float(x) for x in args.keeps.split(",")]:
        m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=args.device,
                                      dtype=DTYPES[args.dtype]).eval()
        if not args.no_trim:
            apply_trimmed_cfg(m)
        if keep < 1.0:
            apply_cfg_interval(m, keep=keep, where=args.where)
        prompt = m.create_voice_clone_prompt(ref_audio=args.ref, ref_text=args.ref_text)
        if args.compile:
            m.llm = torch.compile(m.llm, dynamic=True)
        m.generate(text=texts[0], language="vi", voice_clone_prompt=prompt, num_step=4)
        sync(args.device)

        t0, audio_s = time.time(), 0.0
        for i, t in enumerate(texts):
            torch.manual_seed(1234 + i)
            a = m.generate(text=t, language="vi", voice_clone_prompt=prompt,
                           num_step=args.steps)
            audio_s += len(a[0]) / m.sampling_rate
        sync(args.device)
        total = time.time() - t0
        row = {"keep": keep, "where": args.where, "steps": args.steps,
               "total_seconds": round(total, 2), "audio_seconds": round(audio_s, 2),
               "rtf": round(total / audio_s, 3)}
        rows.append(row)
        print("CFGI " + json.dumps(row), flush=True)
        del m
        if args.device == "mps":
            torch.mps.empty_cache()

    base = rows[0]["total_seconds"]
    print()
    for r in rows:
        print(f"  keep={r['keep']:<5} {r['total_seconds']:>6.2f}s  RTF={r['rtf']:<6} "
              f"nhanh hơn {base / r['total_seconds']:.2f}x")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
