"""Compare one forward pass with and without the trimmed-CFG path.

Isolates the claim ("dropping padded positions is mathematically lossless") from
the diffusion loop, which amplifies any rounding difference into different token
choices.
"""
import argparse, os, sys

import torch


DTYPES = {"fp16": torch.float16, "fp32": torch.float32}

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="mps")
ap.add_argument("--dtype", default="fp16")
ap.add_argument("--ref", required=True)
ap.add_argument("--ref-text", required=True)
args = ap.parse_args()

from omnivoice.models.omnivoice import OmniVoice
from omnivoice_fastpath import apply_trimmed_cfg

m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=args.device,
                              dtype=DTYPES[args.dtype]).eval()
prompt = m.create_voice_clone_prompt(ref_audio=args.ref, ref_text=args.ref_text)

captured = {}
orig = m.forward


def spy(input_ids=None, audio_mask=None, attention_mask=None, **kw):
    if "input_ids" not in captured:
        captured.update(input_ids=input_ids.clone(), audio_mask=audio_mask.clone(),
                        attention_mask=attention_mask.clone())
    return orig(input_ids=input_ids, audio_mask=audio_mask,
                attention_mask=attention_mask, **kw)


m.forward = spy
m.generate(text="Bước đầu tiên là xác định giao tuyến của hai mặt phẳng đã cho.",
           language="vi", voice_clone_prompt=prompt, num_step=2)
m.forward = orig

ids, amask, attn = captured["input_ids"], captured["audio_mask"], captured["attention_mask"]
lengths = attn[:, 0, :, 0].sum(dim=1).tolist()
B = ids.shape[0] // 2
print(f"batch={ids.shape[0]} seq={ids.shape[-1]} effective_lengths={lengths}")

with torch.inference_mode():
    ref = m(input_ids=ids, audio_mask=amask, attention_mask=attn).logits.float().cpu()
    apply_trimmed_cfg(m)
    got = m(input_ids=ids, audio_mask=amask, attention_mask=attn).logits.float().cpu()

# Only positions the sampler actually reads are meaningful.
for row in range(ids.shape[0]):
    length = int(lengths[row])
    a = ref[row, :, :length, :]
    b = got[row, :, :length, :]
    diff = (a - b).abs()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    kind = "cond  " if row < B else "uncond"
    print(f"{kind} row={row} len={length:>4} max_abs_diff={diff.max():.5f} "
          f"mean_abs_diff={diff.mean():.7f} argmax_agreement={100*agree:.2f}%")
