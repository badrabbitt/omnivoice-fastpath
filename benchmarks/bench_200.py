"""Real run of a 200-short-segment job, the way the service would do it.

Sequential calls (batching is a loss on every device measured), one shared
voice clone prompt, per-segment latency recorded.

    python bench_200.py --device mps --steps 16 --ref ref.wav --ref-text "..."
"""
import argparse, json, os, statistics, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sentences import SENTENCES

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--count", type=int, default=200)
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

    t_load = time.time()
    m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=args.device,
                                  dtype=DTYPES[args.dtype]).eval()
    load_s = time.time() - t_load
    if not args.no_trim:
        apply_trimmed_cfg(m)

    t_prompt = time.time()
    prompt = m.create_voice_clone_prompt(ref_audio=args.ref, ref_text=args.ref_text)
    prompt_s = time.time() - t_prompt

    compile_s = 0.0
    if args.compile:
        t0 = time.time()
        m.llm = torch.compile(m.llm, dynamic=True)
        m.generate(text=SENTENCES[0], language="vi", voice_clone_prompt=prompt, num_step=4)
        compile_s = time.time() - t0

    # Segments in the 3-4s range: the shortest sentences of the corpus, cycled.
    short = sorted(SENTENCES, key=len)[:20]
    texts = [short[i % len(short)] for i in range(args.count)]

    m.generate(text=texts[0], language="vi", voice_clone_prompt=prompt, num_step=4)
    if args.device == "cuda":
        torch.cuda.synchronize()

    lat, audio_s = [], 0.0
    t0 = time.time()
    for i, t in enumerate(texts):
        s = time.time()
        a = m.generate(text=t, language="vi", voice_clone_prompt=prompt, num_step=args.steps)
        if args.device == "cuda":
            torch.cuda.synchronize()
        lat.append(time.time() - s)
        audio_s += len(a[0]) / m.sampling_rate
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.count} … {time.time()-t0:.0f}s", flush=True)
    total = time.time() - t0

    out = {
        "device": args.device, "dtype": args.dtype, "steps": args.steps,
        "trimmed_cfg": not args.no_trim, "compiled": args.compile,
        "count": args.count,
        "load_seconds": round(load_s, 1), "prompt_seconds": round(prompt_s, 1),
        "compile_seconds": round(compile_s, 1),
        "generate_seconds": round(total, 1),
        "wall_including_startup": round(load_s + prompt_s + compile_s + total, 1),
        "audio_seconds": round(audio_s, 1),
        "mean_audio_per_segment": round(audio_s / args.count, 2),
        "rtf": round(total / audio_s, 3),
        "sec_per_segment_mean": round(total / args.count, 3),
        "sec_per_segment_median": round(statistics.median(lat), 3),
        "sec_per_segment_p90": round(sorted(lat)[int(0.9 * len(lat))], 3),
    }
    print("RUN200 " + json.dumps(out), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
