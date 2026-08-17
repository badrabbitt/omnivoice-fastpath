"""Paired statistical comparison of TTS configurations.

Reads <root>/<config>/{meta.json,asr.json}, computes per-sentence WER against the
reference text, then compares every config to the baseline with a paired
bootstrap CI and a Wilcoxon signed-rank test.

Run on the host: python eval_stat_report.py [baseline_config]
"""
import json, math, os, random, re, sys, unicodedata
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))
from eval_sentences import SENTENCES

ROOT = Path(os.getenv("EVAL_ROOT_HOST",
                      os.environ.get("OVFP_EVAL_ROOT", "./eval_out")))
BASELINE = sys.argv[1] if len(sys.argv) > 1 else "A_fp32_step32"
random.seed(0)

DIGITS = {"0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
          "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín",
          "10": "mười"}


def norm(s: str) -> list:
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\sÀ-ỹ]", " ", s)
    return [DIGITS.get(w, w) for w in s.split()]


def edits(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[-1]


def load(cfg: str):
    d = ROOT / cfg
    meta = json.loads((d / "meta.json").read_text())
    asr = json.loads((d / "asr.json").read_text())
    wers, ref_words = [], []
    for i, ref in enumerate(SENTENCES):
        key = f"{i:03d}"
        if key not in asr:
            continue
        r, h = norm(ref), norm(asr[key])
        wers.append(edits(r, h) / max(1, len(r)))
        ref_words.append(len(r))
    return meta, wers, ref_words


def bootstrap_ci(diffs: list, reps: int = 20000, alpha: float = 0.05):
    n = len(diffs)
    means = []
    for _ in range(reps):
        means.append(sum(diffs[random.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * reps)]
    hi = means[int((1 - alpha / 2) * reps)]
    return lo, hi


def wilcoxon(diffs: list):
    """Signed-rank test with a normal approximation (ties averaged)."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 6:
        return None, n
    order = sorted(range(n), key=lambda k: abs(nz[k]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(ranks[k] for k in range(n) if nz[k] > 0)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_plus - mu) / sigma
    p = math.erfc(abs(z) / math.sqrt(2))  # two-sided
    return p, n


configs = sorted(d.name for d in ROOT.iterdir()
                 if d.is_dir() and (d / "asr.json").exists())
# Extra args restrict the comparison — auto-voice and voice-cloning runs are not
# comparable to each other, so they must not share a table.
if len(sys.argv) > 2:
    keep = set(sys.argv[2:]) | {BASELINE}
    configs = [c for c in configs if c in keep]
if BASELINE not in configs:
    sys.exit(f"baseline {BASELINE} not found in {configs}")

data = {c: load(c) for c in configs}
base_meta, base_wers, _ = data[BASELINE]
n = len(base_wers)

print(f"Paired evaluation — n={n} sentences, judge=faster-whisper large-v3, "
      f"baseline={BASELINE}\n")
print(f"{'config':<18} {'RTF':>6} {'tổng(s)':>9} {'WER%':>7} {'ΔWER':>7} "
      f"{'CI95 của Δ':>18} {'p':>8}")
print("-" * 78)

for c in configs:
    meta, wers, _ = data[c]
    mean_wer = 100 * sum(wers) / len(wers)
    if c == BASELINE:
        print(f"{c:<18} {meta['rtf']:>6.2f} {meta['total_seconds']:>9.1f} "
              f"{mean_wer:>7.2f} {'—':>7} {'(mốc so sánh)':>18} {'—':>8}")
        continue
    diffs = [100 * (wers[i] - base_wers[i]) for i in range(min(len(wers), n))]
    d_mean = sum(diffs) / len(diffs)
    lo, hi = bootstrap_ci(diffs)
    p, _ = wilcoxon(diffs)
    p_s = "n/a" if p is None else f"{p:.3f}"
    print(f"{c:<18} {meta['rtf']:>6.2f} {meta['total_seconds']:>9.1f} "
          f"{mean_wer:>7.2f} {d_mean:>+7.2f} {f'[{lo:+.2f}, {hi:+.2f}]':>18} {p_s:>8}")

# How many sentences are transcribed perfectly, and how many actually differ
# from the baseline — with WER this low most sentences are ties, which is why
# the Wilcoxon test runs out of signed pairs.
print()
for c in configs:
    _, wers, _ = data[c]
    perfect = sum(1 for w in wers if w == 0)
    if c == BASELINE:
        print(f"{c:<18} {perfect}/{len(wers)} câu khớp hoàn toàn")
        continue
    nz = sum(1 for i in range(min(len(wers), n)) if wers[i] != base_wers[i])
    print(f"{c:<18} {perfect}/{len(wers)} câu khớp hoàn toàn, {nz} câu khác baseline")

# Smallest WER gap this sample size could have detected, from the paired SD.
print()
for c in configs:
    if c == BASELINE:
        continue
    _, wers, _ = data[c]
    diffs = [100 * (wers[i] - base_wers[i]) for i in range(min(len(wers), n))]
    mu = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((d - mu) ** 2 for d in diffs) / max(1, len(diffs) - 1))
    mde = 2.8 * sd / math.sqrt(len(diffs))  # ~80% power, alpha=0.05
    print(f"{c:<18} SD của Δ = {sd:5.2f} điểm → khác biệt nhỏ nhất phát hiện được ≈ {mde:.2f} điểm WER")
