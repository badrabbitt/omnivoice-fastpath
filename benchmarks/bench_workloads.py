"""Benchmark the two production workloads on any device.

  A (short): many short segments generated back to back — measures per-segment
     latency at batch 1 and throughput at larger batch sizes.
  B (long): one very long text — compares OmniVoice's built-in chunking (which
     runs a single item's chunks sequentially) against manual chunking that
     batches independent chunks sharing one voice clone prompt.

    python bench_workloads.py --device cuda --dtype fp16 --steps 32 \
        --mode short,long --batches 1,4,8,16 --ref /path/ref.wav --ref-text "..."
"""
import argparse, json, os, statistics, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sentences import SENTENCES

DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def peak_mem_gb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6


def reset_mem(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def chunk_text(text: str, max_chars: int) -> list:
    """Split on sentence punctuation into pieces of at most max_chars."""
    import re
    parts = re.split(r"(?<=[.!?;:])\s+", text)
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) + 1 > max_chars:
            out.append(cur.strip())
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur.strip():
        out.append(cur.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mode", default="short")
    ap.add_argument("--batches", default="1,4,8,16")
    ap.add_argument("--n-short", type=int, default=16)
    ap.add_argument("--long-minutes", type=float, default=5.0)
    ap.add_argument("--chunk-chars", type=int, default=180)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--sort", action="store_true",
                    help="bucket segments by length before batching")
    ap.add_argument("--flashinfer", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from omnivoice.models.omnivoice import OmniVoice

    t0 = time.time()
    m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=args.device,
                                  dtype=DTYPES[args.dtype]).eval()
    load_s = time.time() - t0

    if args.flashinfer:
        from omnivoice.models.omnivoice_flashinfer import apply_flashinfer
        apply_flashinfer(m)
        print("flashinfer applied", flush=True)

    prompt = None
    if args.ref:
        prompt = m.create_voice_clone_prompt(ref_audio=args.ref, ref_text=args.ref_text)

    def gen(texts, **kw):
        kwargs = dict(text=texts, language="vi" if isinstance(texts, str) else ["vi"] * len(texts),
                      num_step=args.steps, **kw)
        if prompt is not None:
            kwargs["voice_clone_prompt"] = prompt if isinstance(texts, str) else [prompt] * len(texts)
        return m.generate(**kwargs)

    results = {"device": args.device, "dtype": args.dtype, "steps": args.steps,
               "threads": args.threads, "flashinfer": args.flashinfer,
               "load_seconds": round(load_s, 1), "voice_clone": bool(args.ref), "runs": []}

    # Warm up so the first timed run is not paying lazy-init costs.
    gen("Xin chào.")
    sync(args.device)

    modes = args.mode.split(",")

    if "short" in modes:
        pool = (SENTENCES * 4)[: args.n_short]
        if args.sort:
            # Batching pads every item to the longest in the batch, so grouping
            # similar lengths together is what makes batching pay off at all.
            pool = sorted(pool, key=len)
        for b in [int(x) for x in args.batches.split(",")]:
            reset_mem(args.device)
            lat = []
            t0 = time.time()
            audio_s = 0.0
            for i in range(0, len(pool), b):
                group = pool[i:i + b]
                t1 = time.time()
                outs = gen(group[0] if b == 1 else group)
                sync(args.device)
                lat.append(time.time() - t1)
                outs = [outs[0]] if b == 1 else outs
                audio_s += sum(len(o) / m.sampling_rate for o in outs)
            total = time.time() - t0
            row = {"mode": "short", "sorted": args.sort, "batch": b, "segments": len(pool),
                   "total_seconds": round(total, 2), "audio_seconds": round(audio_s, 2),
                   "rtf": round(total / audio_s, 3),
                   "sec_per_segment": round(total / len(pool), 3),
                   "batch_latency_median": round(statistics.median(lat), 2),
                   "peak_mem_gb": round(peak_mem_gb(args.device), 2)}
            results["runs"].append(row)
            print("SHORT " + json.dumps(row), flush=True)

    if "long" in modes:
        # Build a text long enough to cover --long-minutes of speech.
        per_sentence_audio = 4.7
        need = int(args.long_minutes * 60 / per_sentence_audio) + 1
        long_text = " ".join((SENTENCES * (need // len(SENTENCES) + 1))[:need])

        # B1: hand the whole thing to OmniVoice (internal sequential chunking).
        reset_mem(args.device)
        t0 = time.time()
        out = gen(long_text)
        sync(args.device)
        native = time.time() - t0
        native_audio = len(out[0]) / m.sampling_rate
        row = {"mode": "long", "strategy": "native_chunking", "batch": 1,
               "chars": len(long_text), "total_seconds": round(native, 1),
               "audio_seconds": round(native_audio, 1),
               "rtf": round(native / native_audio, 3),
               "peak_mem_gb": round(peak_mem_gb(args.device), 2)}
        results["runs"].append(row)
        print("LONG " + json.dumps(row), flush=True)

        # B2: chunk it ourselves and batch the independent chunks.
        chunks = chunk_text(long_text, args.chunk_chars)
        if args.sort:
            chunks = sorted(chunks, key=len)
        for b in [int(x) for x in args.batches.split(",")]:
            if b == 1:
                continue
            reset_mem(args.device)
            t0 = time.time()
            audio_s = 0.0
            for i in range(0, len(chunks), b):
                group = chunks[i:i + b]
                outs = gen(group if len(group) > 1 else group[0])
                sync(args.device)
                outs = outs if len(group) > 1 else [outs[0]]
                audio_s += sum(len(o) / m.sampling_rate for o in outs)
            total = time.time() - t0
            row = {"mode": "long", "strategy": "manual_chunk_batched", "sorted": args.sort, "batch": b,
                   "chunks": len(chunks), "total_seconds": round(total, 1),
                   "audio_seconds": round(audio_s, 1),
                   "rtf": round(total / audio_s, 3),
                   "peak_mem_gb": round(peak_mem_gb(args.device), 2)}
            results["runs"].append(row)
            print("LONG " + json.dumps(row), flush=True)

    print("RESULTS " + json.dumps(results))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
