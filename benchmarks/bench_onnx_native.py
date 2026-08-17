"""Compare CPU backends for OmniVoice on a native host (macOS/Linux/Windows).

PyTorch fp32 vs ONNX Runtime (fp32 / int8, CPU or CoreML execution provider),
end to end through generate().
"""
import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sentences import SENTENCES


def check_shapes(onnx_path, threads):
    """The exporter can silently specialise the batch dim; find out before use."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    for inp in sess.get_inputs():
        print(f"  input {inp.name}: {inp.shape}")
    results = {}
    for batch, seq in ((2, 200), (1, 200), (2, 320)):
        feeds = {
            "input_ids": np.zeros((batch, 8, seq), dtype=np.int64),
            "audio_mask": np.zeros((batch, seq), dtype=bool),
            "attention_mask": np.ones((batch, 1, seq, seq), dtype=bool),
        }
        try:
            sess.run(None, feeds)
            results[f"batch{batch}_seq{seq}"] = "ok"
        except Exception as exc:
            results[f"batch{batch}_seq{seq}"] = f"FAIL {str(exc)[:120]}"
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--fp32-onnx", default=None)
    ap.add_argument("--int8-onnx", default=None)
    ap.add_argument("--coreml", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    if args.int8_onnx:
        print("int8 graph input shapes:")
        check_shapes(args.int8_onnx, args.threads)
    if args.check_only:
        return

    from omnivoice.models.omnivoice import OmniVoice
    from omnivoice_fastpath import apply_onnx_backend

    texts = sorted(SENTENCES, key=len)[: args.n]
    rows = []

    def run(label, onnx_path=None, provider=None):
        m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu",
                                      dtype=torch.float32).eval()
        if onnx_path:
            if provider:
                import onnxruntime as ort
                from omnivoice.models.omnivoice import OmniVoiceModelOutput

                so = ort.SessionOptions()
                so.intra_op_num_threads = args.threads
                sess = ort.InferenceSession(onnx_path, so, providers=[provider])

                def fwd(input_ids=None, audio_mask=None, attention_mask=None, **kw):
                    logits = sess.run(None, {
                        "input_ids": input_ids.cpu().numpy(),
                        "audio_mask": audio_mask.cpu().numpy(),
                        "attention_mask": attention_mask.cpu().numpy()})[0]
                    return OmniVoiceModelOutput(loss=None, logits=torch.from_numpy(logits))

                m.forward = fwd
            else:
                apply_onnx_backend(m, onnx_path, threads=args.threads)
        prompt = m.create_voice_clone_prompt(ref_audio=args.ref, ref_text=args.ref_text)
        m.generate(text=texts[0], language="vi", voice_clone_prompt=prompt, num_step=4)
        t0, audio_s = time.time(), 0.0
        for i, t in enumerate(texts):
            torch.manual_seed(1234 + i)
            a = m.generate(text=t, language="vi", voice_clone_prompt=prompt,
                           num_step=args.steps)
            audio_s += len(a[0]) / m.sampling_rate
        total = time.time() - t0
        row = {"backend": label, "total_seconds": round(total, 2),
               "audio_seconds": round(audio_s, 2), "rtf": round(total / audio_s, 3)}
        rows.append(row)
        print("NATIVE " + json.dumps(row), flush=True)
        del m

    run("pytorch_fp32")
    if args.int8_onnx:
        try:
            run("onnx_int8_cpu", args.int8_onnx)
        except Exception as exc:
            print(f"NATIVE onnx_int8_cpu FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    if args.fp32_onnx:
        try:
            run("onnx_fp32_cpu", args.fp32_onnx)
        except Exception as exc:
            print(f"NATIVE onnx_fp32_cpu FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    if args.coreml and args.fp32_onnx:
        try:
            run("onnx_fp32_coreml", args.fp32_onnx, provider="CoreMLExecutionProvider")
        except Exception as exc:
            print(f"NATIVE onnx_fp32_coreml FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)

    if rows:
        base = rows[0]["total_seconds"]
        print()
        for r in rows:
            print(f"  {r['backend']:<20} {r['total_seconds']:>7.2f}s  RTF={r['rtf']:<6} "
                  f"so với PyTorch {base / r['total_seconds']:.2f}x")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
