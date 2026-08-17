"""Compare the tuned stack against upstream OmniVoice as it ships.

Three baselines on the same machine and the same sentences:

  A. upstream CLI      — `omnivoice-infer` per segment, exactly as the README
                         shows. Reloads the model on every call.
  B. upstream API      — same defaults (fp16, 32 steps, plain forward) but with
                         the model loaded once. Isolates the serving fix from
                         the inference tuning.
  C. tuned stack       — resident model + trimmed CFG + torch.compile +
                         12 steps + 3s reference.

B vs C is the honest "did we beat upstream" number; A vs C is what someone
following the README would actually feel.
"""
import argparse, json, os, subprocess, sys, tempfile, time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sentences import SENTENCES


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--cli", default="omnivoice-infer")
    ap.add_argument("--long-ref", required=True, help="reference a user would grab (~6s)")
    ap.add_argument("--long-ref-text", required=True)
    ap.add_argument("--short-ref", required=True, help="trimmed reference (~3s)")
    ap.add_argument("--short-ref-text", required=True)
    ap.add_argument("--skip-cli", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    texts = sorted(SENTENCES, key=len)[: args.n]
    rows = []

    if not args.skip_cli:
        tmp = tempfile.mkdtemp(prefix="upstream-cli-")
        t0 = time.time()
        for i, text in enumerate(texts):
            subprocess.run([args.cli, "--text", text, "--output", f"{tmp}/{i}.wav",
                            "--language", "vi", "--ref_audio", args.long_ref,
                            "--ref_text", args.long_ref_text],
                           check=True, capture_output=True)
        total = time.time() - t0
        import wave
        audio_s = 0.0
        for i in range(len(texts)):
            w = wave.open(f"{tmp}/{i}.wav")
            audio_s += w.getnframes() / w.getframerate()
        rows.append({"config": "A. upstream CLI (mỗi đoạn một tiến trình)",
                     "total_seconds": round(total, 2), "audio_seconds": round(audio_s, 2),
                     "rtf": round(total / audio_s, 3),
                     "sec_per_segment": round(total / len(texts), 2)})
        print("VS " + json.dumps(rows[-1], ensure_ascii=False), flush=True)

    from omnivoice.models.omnivoice import OmniVoice
    from omnivoice_fastpath import apply_trimmed_cfg

    def measure(label, ref, ref_text, steps, tuned):
        m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=args.device,
                                      dtype=torch.float16).eval()
        if tuned:
            apply_trimmed_cfg(m)
        prompt = m.create_voice_clone_prompt(ref_audio=ref, ref_text=ref_text)
        if tuned:
            m.llm = torch.compile(m.llm, dynamic=True)
            m.generate(text=texts[0], language="vi", voice_clone_prompt=prompt, num_step=4)
        m.generate(text=texts[0], language="vi", voice_clone_prompt=prompt, num_step=4)
        sync(args.device)
        t0, audio_s = time.time(), 0.0
        for i, text in enumerate(texts):
            torch.manual_seed(1234 + i)
            a = m.generate(text=text, language="vi", voice_clone_prompt=prompt, num_step=steps)
            audio_s += len(a[0]) / m.sampling_rate
        sync(args.device)
        total = time.time() - t0
        row = {"config": label, "total_seconds": round(total, 2),
               "audio_seconds": round(audio_s, 2), "rtf": round(total / audio_s, 3),
               "sec_per_segment": round(total / len(texts), 2)}
        rows.append(row)
        print("VS " + json.dumps(row, ensure_ascii=False), flush=True)
        del m

    measure("B. upstream API (model thường trú, 32 bước, ref 6s)",
            args.long_ref, args.long_ref_text, 32, tuned=False)
    measure("C. bản tối ưu (trimmed CFG + compile + 12 bước + ref 3s)",
            args.short_ref, args.short_ref_text, 12, tuned=True)

    print()
    tuned = rows[-1]["total_seconds"]
    for r in rows:
        print(f"  {r['config']:<52} {r['total_seconds']:>7.2f}s  RTF={r['rtf']:<6} "
              f"{r['sec_per_segment']:>5.2f}s/đoạn  nhanh hơn {r['total_seconds'] / tuned:.2f}x")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
