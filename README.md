# omnivoice-fastpath

Inference speedups for [OmniVoice](https://github.com/k2-fsa/OmniVoice) TTS, with the
measurements behind them. 4.6x faster than the stock Python API on the same
hardware, at no quality difference we could detect over 48 paired sentences.

Half of this repo is negative results. Batching, ONNX fp32, CUDA graphs and CFG
interval all sound like they should help. None of them do, and the numbers
explaining why are more useful than another list of tricks.

## Numbers

Apple M2 Max, MPS, fp16, 10 voice-cloning segments of 3-4s each.

| Setup | Total | RTF | Per segment |
|---|---|---|---|
| `omnivoice-infer` CLI, one process per segment | 112.5s | 3.36 | 11.25s |
| Python API, model resident, 32 steps, 6s reference | 60.3s | 1.81 | 6.03s |
| **This repo** — trimmed CFG + compile + 12 steps + 3s reference | **13.1s** | **0.41** | **1.31s** |

The middle row is the honest baseline: 4.6x. The top row is 8.6x, but a third of
that gap is just not reloading a 813M-parameter model for every sentence, which
any serving setup fixes on its own.

Where the 4.6x comes from:

| Lever | Speedup | Cost |
|---|---|---|
| 32 -> 12 diffusion steps | 2.34x | none we could measure |
| 6s -> 3s reference clip | 1.37x | none we could measure |
| Trimmed CFG | 1.18x on MPS, 1.42x on CPU | none, it is arithmetically lossless |
| `torch.compile` | 1.09x on MPS, 1.42x on CUDA | 10s startup on MPS, 137-218s on CUDA |

Multiplied out that is 4.12x; measured is 4.61x, because `torch.compile` pays off
more once the other levers have made each step cheaper.

## Install

```bash
pip install omnivoice omnivoice-fastpath
```

Or from source:

```bash
git clone https://github.com/badrabbitt/omnivoice-fastpath
cd omnivoice-fastpath && pip install -e .
```

## Use

```python
import torch
from omnivoice.models.omnivoice import OmniVoice
from omnivoice_fastpath import apply_trimmed_cfg

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="mps", dtype=torch.float16)
apply_trimmed_cfg(model)                  # skip on CUDA, see below
model.llm = torch.compile(model.llm, dynamic=True)

prompt = model.create_voice_clone_prompt(ref_audio="ref_3s.wav", ref_text="...")
audio = model.generate(text="...", language="vi", voice_clone_prompt=prompt, num_step=12)
```

Reuse `prompt` across segments. Encoding a reference costs 1-2s, and a dubbing job
is usually hundreds of segments sharing one voice.

## Trimmed CFG

The one thing here that is not just parameter choice.

Every diffusion step runs a single forward over `2B` rows: `B` conditional rows of
length `c_len` (reference + text + target) and `B` unconditional rows of length
`u_len` (target only). Both halves sit in one tensor, so the shorter unconditional
rows get padded up to `max_c_len`. In voice-cloning mode we measured `c_len=295`
against `u_len=89` — the unconditional half was doing 3.3x more work than it needed
to, on every step.

Those padded positions attend only to themselves, and no real position attends to
them, so dropping them cannot change the result. Verified at the logits level
rather than argued: on CPU/fp32 the maximum absolute difference against the
unpatched model is **exactly 0.00000** with 100% argmax agreement. Under fp16 the
difference is 0.22 with 98.4% agreement, which is GEMM reduction order changing
with the tensor shape, not a different computation.

```python
from omnivoice_fastpath import apply_trimmed_cfg
apply_trimmed_cfg(model)
```

It wraps `forward`, so upstream source stays untouched.

**Turn it off on CUDA.** Fewer tokens but twice the kernel launches: that trade wins
where compute is the bottleneck and loses where launch latency is.

| Device | With trimmed CFG |
|---|---|
| CPU (fp32) | 1.42x faster |
| MPS (fp16) | 1.18x faster |
| CUDA T4 (fp16) | **0.64x — slower** |

## Steps

12 is the floor. 8 breaks.

Paired evaluation, 48 sentences, same seed per sentence across configs, judged by
faster-whisper large-v3, 95% CI from a paired bootstrap over per-sentence WER.

| Steps | RTF | WER | Δ vs 32 steps | 95% CI | p |
|---|---|---|---|---|---|
| 32 | 1.01 | 3.38% | — | — | — |
| 16 | 0.53 | 3.14% | −0.24 | [−1.63, +1.24] | 0.695 |
| 12 | 0.43 | 2.51% | −0.87 | [−2.20, +0.52] | 0.331 |
| 8 | 0.32 | 6.45% | **+3.07** | **[+1.04, +5.40]** | **0.009** |

12 steps is indistinguishable from 32 and 2.34x faster. Do not read 2.51% as *better*
than 32 steps — the interval spans zero, that is noise. At 8 steps WER nearly doubles
and sentences transcribed perfectly drop from 32/48 to 24/48.

## Reference length

A reference clip is prepended to the sequence on every step, so its length is
multiplied by the step count. Natural clips, each with its exact transcript:

| Reference | Tokens | Mean seq len | Total | Speedup |
|---|---|---|---|---|
| 3.1s | 74 | 227.6 | 31.3s | 1.37x |
| 4.0s | 96 | 249.1 | 36.4s | 1.18x |
| 5.2s | 127 | 285.2 | 40.9s | 1.05x |
| 6.2s | 149 | 314.0 | 42.9s | — |

Cutting 6.2s to 3.2s of the same speaker, with the transcript cut at the matching
word boundary, showed no WER difference across 48 sentences (Δ −0.02, CI
[−0.95, +0.92], p=0.754). WER does not measure timbre similarity, so listen before
committing to a short reference for a voice you care about.

One trap: if you trim the audio and guess where to cut the transcript, the model's
pacing estimate goes wrong and generates longer audio, which inflates RTF and makes
the speedup look better than it is. Our first pass reported 1.86x for this reason.
Cut on a word boundary and use the exact transcript.

## What did not work

| Idea | Result |
|---|---|
| Batching segments | Slower everywhere. T4 RTF 0.446 at batch 1 -> 0.584 at batch 32. Sorting by length first does not help. |
| ONNX Runtime fp32 | Bit-identical to PyTorch, 2.7% faster per forward, net slower end to end once tensors round-trip through numpy. |
| ONNX Runtime int8 | Genuinely 1.54x on macOS CPU, 2.3x on Linux CPU — but costs about 1 WER point, and CPU is 5x slower than the same machine's GPU anyway. |
| CUDA graphs (`mode="reduce-overhead"`) | 8% slower than default compile and 2.8x the compile time. Sequence length varies per sentence, so inductor re-records the graph per shape. |
| CFG interval | 1.21x, but +2.18 WER points (CI [+0.74, +3.67], p=0.009). Dropping the early half is just as bad as the late half; guidance matters throughout. |
| CoreML execution provider | 1.10x. Falls back to CPU for most of this model's ops. |
| FlashInfer on T4 | `KeyError: 'sm_75'`. Its kernels need Ampere or newer. |

Batching deserves a note, since it is the first thing everyone tries. Two reasons it
fails: CFG already doubles the batch, so a single item saturates the device; and
`_generate_iterative` scores tokens in a per-item Python loop each step, so batch 32
over 32 steps means 1024 small serialized operations. Making batching pay would mean
vectorizing that loop upstream.

## Devices

32 short voice-cloning segments, 32 steps, batch 1.

| Device | RTF | Per segment | Memory |
|---|---|---|---|
| Colab T4, fp16 | 0.45 | 1.79s | 2.3GB VRAM |
| M2 Max MPS, fp16 | 1.37 | 5.52s | 4.4GB |
| Docker CPU, fp32 | 9.85 | 39.97s | 4.7GB |

Voice cloning is much more expensive than auto-voice mode and the bill lands almost
entirely on CPU: the same sentences run at RTF 4.39 without a reference and 9.85 with
one, because the reference lengthens the sequence and attention is quadratic. Any RTF
you see quoted for auto-voice is optimistic for real dubbing.

Long-form is cheaper per second than short segments on every device (T4: 0.33 vs 0.45)
because each internal chunk is ~8s, so fixed per-call cost amortizes. Memory stays
flat with length, so hours-long narration will not OOM. Hand the whole text to one
`generate()` call and let OmniVoice chunk it — chunking it yourself and batching the
pieces is slower.

## Reproducing

```bash
pip install -e ".[onnx,eval]"

# headline comparison
python benchmarks/bench_vs_upstream.py --device mps --n 10 \
  --long-ref refs/ref_full.wav --long-ref-text "..." \
  --short-ref refs/ref_3s.wav --short-ref-text "..."

# cumulative effect of each lever
python benchmarks/bench_stack.py --device mps --long-ref ... --short-ref ...

# prove trimmed CFG changes nothing, at the logits level
python benchmarks/check_fastpath_parity.py --device cpu --dtype fp32 --ref ... --ref-text "..."
```

Quality evaluation is three steps: generate a set per config, transcribe with
faster-whisper large-v3, then compare paired.

```bash
python eval/eval_stat_gen.py torch A_step32 32
python eval/eval_stat_asr.py A_step32 B_step12
python eval/eval_stat_report.py A_step32
```

The corpus is 48 Vietnamese sentences from a maths-explainer domain in
`benchmarks/eval_sentences.py`. Swap it for your own language and content; the
speed results are language-independent but the WER numbers are not.

## Method notes

Everything above is measured on one machine (M2 Max, 12 CPU, 64GB; Docker limited to
12 CPU / 12GB) and one Colab T4. Numbers will move on other hardware, and the
CPU/GPU trade-offs behind trimmed CFG and batching could invert on a different
device.

Two things worth knowing if you extend this work.

The ASR judge matters more than the model under test. An earlier version of this
evaluation used faster-whisper `base` and reported 20-33% WER with a completely
different ranking of the configurations. large-v3 puts the same audio at 2-3%. If
your judge is worse than your system, you are measuring the judge.

At these WER levels most sentences are ties, so the interesting statistic is how many
sentences differ at all — 12 steps differs from 32 on 12 of 48 sentences, and only
those carry any signal. The minimum difference this design can detect is about 1.3
WER points, so treat anything smaller as unresolved rather than equal.

## Still on the table

The prefix — reference audio plus text — is 61-64% of the sequence and is completely
static across every step, yet gets recomputed each time. Caching its KV states is what
[dLLM-Cache](https://arxiv.org/abs/2506.06295), [Fast-dLLM](https://arxiv.org/abs/2505.22618)
and [Elastic-Cache](https://arxiv.org/pdf/2510.14973) do for diffusion LLMs, all
training-free. Counting tokens puts the ceiling here around 1.8x. It is an
approximation — the prefix attends to the target bidirectionally, so freezing its KV
changes the result — and nobody has published this for diffusion TTS specifically.

[Guidance distillation](https://arxiv.org/pdf/2210.03142) removes the unconditional
forward entirely for up to 2x, but needs fine-tuning.

## Credits

OmniVoice is by the [k2-fsa team](https://github.com/k2-fsa/OmniVoice), Apache-2.0.
This repo adds nothing to the model — it wraps the published API and measures it.
Benchmarks ran against 0.2.1 (commit `38e992b`).

Apache-2.0.
