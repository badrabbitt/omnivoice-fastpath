# Raw measurements

Every number in the README, with the conditions attached. Hardware: MacBook M2 Max
(12 CPU, 64GB), Docker Desktop given 12 CPU / 12GB, Colab T4 (sm_75, 15GB).
OmniVoice 0.2.1 (`38e992b`) throughout.

## Against upstream

MPS, fp16, 10 voice-cloning segments of 3-4s.

| Setup | Total | RTF | Per segment |
|---|---|---|---|
| `omnivoice-infer` CLI, process per segment | 112.53s | 3.357 | 11.25s |
| Python API resident, 32 steps, 6.2s reference | 60.29s | 1.811 | 6.03s |
| Trimmed CFG + compile + 12 steps + 3.2s reference | 13.08s | 0.413 | 1.31s |

## Cumulative levers

MPS, fp16, 8 sentences, voice cloning.

| Config | Total | RTF | Cumulative |
|---|---|---|---|
| Baseline (6.2s ref, 32 steps) | 42.63s | 1.228 | 1.00x |
| + trimmed CFG | 36.01s | 1.036 | 1.18x |
| + torch.compile | 33.60s | 0.972 | 1.27x |
| + 3.1s reference | 28.57s | 0.867 | 1.49x |
| + 16 steps | 15.24s | 0.460 | 2.80x |

T4, fp16, same sentences, without trimmed CFG:

| Config | Total | RTF | Cumulative |
|---|---|---|---|
| Baseline (6.2s ref, 32 steps) | 14.25s | 0.411 | 1.00x |
| + torch.compile | 10.04s | 0.291 | 1.42x |
| + 3.1s reference | 7.98s | 0.245 | 1.78x |
| + 16 steps | 4.28s | 0.132 | 3.33x |

Docker CPU, fp32, 8 threads, 6 sentences:

| Config | Total | RTF | Cumulative |
|---|---|---|---|
| Baseline (6.2s ref, 32 steps) | 265.55s | 9.92 | 1.00x |
| + trimmed CFG | 186.94s | 6.98 | 1.42x |
| + 3.1s reference | 148.92s | 5.85 | 1.78x |
| + 16 steps | 77.03s | 3.07 | 3.45x |

## Trimmed CFG parity

One forward, same inputs, patched against unpatched. Sequence 295, effective
lengths [295, 89].

| Device / dtype | Row | Max abs diff | Mean abs diff | Argmax agreement |
|---|---|---|---|---|
| CPU fp32 | cond | 0.00000 | 0.0000000 | 100.00% |
| CPU fp32 | uncond | 0.00008 | 0.0000067 | 100.00% |
| MPS fp16 | cond | 0.21875 | 0.0118977 | 98.39% |
| MPS fp16 | uncond | 0.12500 | 0.0032315 | 100.00% |

## Quality, paired, n=48

faster-whisper large-v3, same seed per sentence index, 95% CI from paired bootstrap
(20000 resamples), Wilcoxon signed-rank where enough discordant pairs exist.

Voice cloning, 3.2s reference, varying steps (baseline 32):

| Steps | RTF | WER | ΔWER | CI95 | p | Perfect | Differ |
|---|---|---|---|---|---|---|---|
| 32 | 1.01 | 3.38% | — | — | — | 32/48 | — |
| 16 | 0.53 | 3.14% | −0.24 | [−1.63, +1.24] | 0.695 | 33/48 | 12 |
| 12 | 0.43 | 2.51% | −0.87 | [−2.20, +0.52] | 0.331 | 36/48 | 14 |
| 8 | 0.32 | 6.45% | +3.07 | [+1.04, +5.40] | 0.009 | 24/48 | 21 |

Reference length, same speaker, 32 steps:

| Reference | RTF | WER | ΔWER | CI95 | p |
|---|---|---|---|---|---|
| 6.17s | 1.13 | 3.40% | — | — | — |
| 3.22s | 1.01 | 3.38% | −0.02 | [−0.95, +0.92] | 0.754 |

CFG interval, 12 steps (baseline full guidance):

| Config | RTF | WER | ΔWER | CI95 | p |
|---|---|---|---|---|---|
| Full | 0.43 | 2.51% | — | — | — |
| keep 0.75, drop late | 0.40 | 3.84% | +1.33 | [+0.23, +2.55] | 0.083 |
| keep 0.50, drop late | 0.36 | 4.69% | +2.18 | [+0.74, +3.67] | 0.009 |
| keep 0.50, drop early | — | 4.48% | +1.97 | [+0.74, +3.28] | 0.026 |

Auto-voice mode, fp32 CPU, varying backend (baseline 32 steps fp32):

| Config | RTF | WER | ΔWER | CI95 | p |
|---|---|---|---|---|---|
| fp32, 32 steps | 4.39 | 2.17% | — | — | — |
| fp32, 16 steps | 2.55 | 2.99% | +0.82 | [+0.13, +1.78] | — |
| ONNX int8, 32 steps | 2.23 | 3.16% | +0.99 | [+0.16, +1.89] | 0.050 |
| ONNX int8, 16 steps | 0.99 | 3.41% | +1.25 | [−0.04, +2.65] | 0.126 |

The 16-step penalty here (+0.82) and in the voice-cloning table (−0.24) disagree.
Both intervals touch zero, so the safe reading is that 16 steps sits somewhere
between "no difference" and "about 0.8 points worse", not that either figure is
settled.

## Batching

32 segments, voice cloning, 32 steps.

| Batch | T4 RTF | MPS RTF |
|---|---|---|
| 1 | 0.446 | 1.368 |
| 4 | 0.475 | 1.267 |
| 8 | 0.518 | 1.371 |
| 16 | 0.547 | — |
| 32 | 0.584 | — |

Sorting by length before batching, T4: 0.491 at batch 1 rising to 0.626 at batch 32.

## Long form

5 minutes of continuous audio.

| Device / strategy | Total | RTF | Memory |
|---|---|---|---|
| T4, built-in chunking, 32 steps | 82.8s | 0.326 | 2.26GB |
| T4, manual chunks batched by 4 | 105.5s | 0.413 | 2.72GB |
| T4, manual chunks batched by 8 | 108.2s | 0.424 | 3.41GB |
| MPS, built-in chunking, 32 steps | 208.0s | 0.816 | 4.42GB |
| MPS, built-in chunking, 16 steps | 110.5s | 0.434 | 4.58GB |

## ONNX Runtime

Single forward, Docker CPU aarch64:

| Backend | Median | Argmax agreement vs PyTorch |
|---|---|---|
| PyTorch fp32 | 0.642s | — |
| ORT fp32 | 0.625s | 100.00% |
| ORT int8, all MatMul | 0.314s | 51.43% |
| ORT int8, audio head kept fp32 | 0.266s | 76.66% |

Excluding one node — the `[1024, 8200]` audio-head projection — recovers most of the
agreement at no speed cost. Quantizing it puts noise straight into token selection.

End to end, macOS native, 12 steps, 6 sentences:

| Backend | Total | RTF | vs PyTorch |
|---|---|---|---|
| PyTorch fp32 | 58.29s | 3.151 | 1.00x |
| ONNX int8 CPU | 37.83s | 2.049 | 1.54x |
| ONNX fp32 CPU | 54.54s | 2.948 | 1.07x |
| ONNX fp32 CoreML | 52.85s | 2.857 | 1.10x |

## CUDA graphs

T4, 16 steps, 3.2s reference, 10 sentences:

| Config | Total | RTF | Compile |
|---|---|---|---|
| No compile | 7.73s | 0.245 | 0s |
| compile, default mode | 4.42s | 0.140 | 137s |
| compile, `reduce-overhead` | 4.78s | 0.151 | 384s |
| trimmed CFG + `reduce-overhead` | 5.72s | 0.181 | 660s |

## Where the time goes

MPS, 16 steps, 3.2s reference, trimmed CFG + compile: 82.8% of wall time is inside
the transformer forward (85.3ms per step), 17.2% is the per-item scoring loop,
decode and post-processing. That fixed share does not shrink with step count, which
is why 8 steps is only 3.12x faster than 32 rather than 4x.

Prefix share of the sequence, 3.2s reference, 3-4s segments:

| Total | Target | Prefix | Prefix share |
|---|---|---|---|
| 192-200 | 69-78 | 119-123 | 61-64% |

## Whole stack against a clean upstream baseline

The comparison the component evaluations could not substitute for. Upstream
defaults with no wrappers at all, against everything this repo recommends.

| Config | RTF | Total | WER | ΔWER | CI95 | p | Clean |
|---|---|---|---|---|---|---|---|
| Upstream, 32 steps, 6.2s ref | 1.31 | 250.2s | 3.00% | — | — | — | 35/48 |
| Tuned, 12 steps, 3.2s ref | 0.36 | 65.0s | 4.38% | +1.38 | [−0.02, +2.79] | 0.081 | 28/48 |

SD of the paired difference is 5.00 points, so this design detects about 2.02
points. The components predicted −0.9 (−0.87 from steps, −0.02 from reference
length); the direct measurement says +1.38. Both readings are inside the noise
floor of their own comparisons, which is the point: at this sample size the
evaluation cannot resolve effects of this size, and stacking separate
"no detectable difference" results into a claim about the combination was wrong.
