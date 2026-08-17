"""Where does a diffusion step actually spend its time?

Splits generate() into the transformer forward versus everything else (the
per-item scoring loop: log_softmax, gumbel, topk, masked_fill, plus decode and
post-processing). If "other" is a large share, the forward is no longer the
thing worth optimising.
"""
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
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from omnivoice.models.omnivoice import OmniVoice
    try:
        from omnivoice_fastpath import apply_trimmed_cfg
    except ImportError:
        from omnivoice_fastpath import apply_trimmed_cfg

    m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=args.device,
                                  dtype=DTYPES[args.dtype]).eval()
    if not args.no_trim:
        apply_trimmed_cfg(m)
    prompt = m.create_voice_clone_prompt(ref_audio=args.ref, ref_text=args.ref_text)

    if args.compile:
        m.llm = torch.compile(m.llm, dynamic=True)
        m.generate(text=SENTENCES[0], language="vi", voice_clone_prompt=prompt, num_step=4)

    stats = {"forward_seconds": 0.0, "forward_calls": 0}
    inner = m.forward

    def timed(*a, **kw):
        sync(args.device)
        t0 = time.perf_counter()
        out = inner(*a, **kw)
        sync(args.device)
        stats["forward_seconds"] += time.perf_counter() - t0
        stats["forward_calls"] += 1
        return out

    m.forward = timed

    texts = sorted(SENTENCES, key=len)[: args.n]
    m.generate(text=texts[0], language="vi", voice_clone_prompt=prompt, num_step=4)
    stats["forward_seconds"] = 0.0
    stats["forward_calls"] = 0

    sync(args.device)
    t0 = time.time()
    audio_s = 0.0
    for t in texts:
        a = m.generate(text=t, language="vi", voice_clone_prompt=prompt, num_step=args.steps)
        audio_s += len(a[0]) / m.sampling_rate
    sync(args.device)
    total = time.time() - t0

    fwd = stats["forward_seconds"]
    out = {"device": args.device, "steps": args.steps, "segments": len(texts),
           "trimmed_cfg": not args.no_trim, "compiled": args.compile,
           "total_seconds": round(total, 2),
           "forward_seconds": round(fwd, 2),
           "other_seconds": round(total - fwd, 2),
           "forward_share_pct": round(100 * fwd / total, 1),
           "forward_calls": stats["forward_calls"],
           "ms_per_forward": round(1000 * fwd / max(1, stats["forward_calls"]), 2),
           "audio_seconds": round(audio_s, 2),
           "rtf": round(total / audio_s, 3)}
    print("PROFILE " + json.dumps(out), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
