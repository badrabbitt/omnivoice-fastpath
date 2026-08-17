"""Stack the speedups and measure each one's marginal contribution.

Configurations are cumulative, so each row shows what the next lever adds on top
of everything before it.

    python bench_stack.py --device mps --dtype fp16
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sentences import SENTENCES

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--long-ref", required=True, help="~6s reference (baseline)")
    ap.add_argument("--long-ref-id", type=int, default=0)
    ap.add_argument("--short-ref", required=True, help="~3s reference")
    ap.add_argument("--short-ref-id", type=int, default=12)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    from omnivoice.models.omnivoice import OmniVoice
    from omnivoice_fastpath import apply_trimmed_cfg

    m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=args.device,
                                  dtype=DTYPES[args.dtype]).eval()
    long_prompt = m.create_voice_clone_prompt(ref_audio=args.long_ref,
                                              ref_text=SENTENCES[args.long_ref_id])
    short_prompt = m.create_voice_clone_prompt(ref_audio=args.short_ref,
                                               ref_text=SENTENCES[args.short_ref_id])
    texts = SENTENCES[: args.n]

    def measure(label, prompt, steps):
        m.generate(text=texts[0], language="vi", voice_clone_prompt=prompt, num_step=4)
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0, audio_s = time.time(), 0.0
        for i, t in enumerate(texts):
            torch.manual_seed(1234 + i)
            a = m.generate(text=t, language="vi", voice_clone_prompt=prompt, num_step=steps)
            audio_s += len(a[0]) / m.sampling_rate
        if args.device == "cuda":
            torch.cuda.synchronize()
        total = time.time() - t0
        row = {"config": label, "steps": steps, "total_seconds": round(total, 2),
               "audio_seconds": round(audio_s, 2), "rtf": round(total / audio_s, 3)}
        print("STACK " + json.dumps(row, ensure_ascii=False), flush=True)
        return row

    rows = [measure("1. gốc (ref ~6s, 32 bước)", long_prompt, 32)]

    apply_trimmed_cfg(m)
    rows.append(measure("2. + trimmed CFG", long_prompt, 32))

    if not args.no_compile:
        t0 = time.time()
        m.llm = torch.compile(m.llm, dynamic=True)
        m.generate(text=texts[0], language="vi", voice_clone_prompt=long_prompt, num_step=4)
        print(f"(compile + warmup {time.time() - t0:.1f}s, trả một lần lúc khởi động)", flush=True)
        rows.append(measure("3. + torch.compile", long_prompt, 32))

    rows.append(measure("4. + ref ~3s", short_prompt, 32))
    rows.append(measure("5. + 16 bước", short_prompt, 16))

    base = rows[0]["total_seconds"]
    print()
    for r in rows:
        print(f"  {r['config']:<32} {r['total_seconds']:>7.2f}s  RTF={r['rtf']:<6} "
              f"cộng dồn {base / r['total_seconds']:.2f}x")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"device": args.device, "dtype": args.dtype, "rows": rows}, f,
                      ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
