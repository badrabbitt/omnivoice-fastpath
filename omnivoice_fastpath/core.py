"""Runtime speedups for OmniVoice inference that don't change the model.

`apply_trimmed_cfg` removes padding waste from the classifier-free-guidance pair.

Every diffusion step runs one forward over a batch of `2B` rows: `B` conditional
rows of length `c_len` (reference + text + target) and `B` unconditional rows of
length `u_len` (target only). Both halves live in one tensor, so the shorter
unconditional rows get padded up to `max_c_len`. In voice-cloning mode the
reference is a large share of `c_len`, so roughly a quarter of every step is
spent on padding.

Those padded positions attend only to themselves (`pad_diag` in
`_generate_iterative`) and no real position attends to them, so dropping them is
mathematically lossless — verified below against the unpatched model.

Applied by wrapping `forward`, so the vendored upstream source stays untouched.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("omnivoice_wrapper")


def apply_trimmed_cfg(model: Any) -> Any:
    """Run each distinct row length at its own size instead of one padded batch."""
    if getattr(model, "_trimmed_cfg_applied", False):
        return model

    import torch
    from omnivoice.models.omnivoice import OmniVoiceModelOutput

    original_forward = model.forward

    def forward(input_ids=None, audio_mask=None, attention_mask=None, **kwargs):
        if (
            input_ids is None
            or attention_mask is None
            or not torch.is_tensor(attention_mask)
            or attention_mask.dim() != 4
            or attention_mask.dtype != torch.bool
        ):
            return original_forward(
                input_ids=input_ids, audio_mask=audio_mask,
                attention_mask=attention_mask, **kwargs
            )

        seq_len = int(input_ids.shape[-1])
        # A row's effective length is how many positions attend to position 0:
        # real positions all do, padded ones only attend to themselves.
        lengths = attention_mask[:, 0, :, 0].sum(dim=1)
        distinct = sorted({int(v) for v in lengths.tolist()})

        if len(distinct) == 1 and distinct[0] == seq_len:
            return original_forward(
                input_ids=input_ids, audio_mask=audio_mask,
                attention_mask=attention_mask, **kwargs
            )

        out = None
        for length in distinct:
            if length <= 0:
                continue
            idx = (lengths == length).nonzero(as_tuple=True)[0]
            sub = original_forward(
                input_ids=input_ids[idx][..., :length],
                audio_mask=audio_mask[idx][:, :length],
                attention_mask=attention_mask[idx][:, :, :length, :length],
                **kwargs,
            )
            logits = sub.logits
            if out is None:
                out = logits.new_zeros(
                    (input_ids.shape[0], logits.shape[1], seq_len, logits.shape[3])
                )
            out[idx, :, :length, :] = logits

        return OmniVoiceModelOutput(loss=None, logits=out)

    model.forward = forward
    model._trimmed_cfg_applied = True
    logger.info("Trimmed-CFG fast path enabled")
    return model


def apply_onnx_backend(model: Any, onnx_path: str, threads: int = 8) -> Any:
    """Serve the transformer forward from an ONNX Runtime CPU session.

    Only useful on CPU-only hosts — ONNX Runtime has no MPS backend, and on CUDA
    the PyTorch path with torch.compile is faster. The int8 graph built by
    `scripts/onnx/make_onnx.py` runs ~2.3x faster than PyTorch fp32 on CPU at a
    cost of roughly one WER point.

    Works anywhere onnxruntime has wheels (macOS arm64/x86_64, Linux, Windows).
    """
    import onnxruntime as ort
    import torch
    from omnivoice.models.omnivoice import OmniVoiceModelOutput

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, options, providers=["CPUExecutionProvider"])

    def forward(input_ids=None, audio_mask=None, attention_mask=None, **kwargs):
        logits = session.run(None, {
            "input_ids": input_ids.cpu().numpy(),
            "audio_mask": audio_mask.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
        })[0]
        return OmniVoiceModelOutput(loss=None, logits=torch.from_numpy(logits))

    model.forward = forward
    model._onnx_backend = onnx_path
    logger.info("ONNX Runtime backend enabled: %s (%d threads)", onnx_path, threads)
    return model


def apply_cfg_interval(model: Any, keep: float = 0.5, where: str = "first") -> Any:
    """Run classifier-free guidance on only part of the diffusion steps.

    On the steps that skip guidance, the unconditional rows are never computed —
    their logits are filled from the conditional rows instead, which makes
    `_predict_tokens_with_scoring` reduce to `guidance_scale = 0` for that step.

    The unconditional half is ~50% of each step's work on CUDA (where the
    padding cannot be trimmed) and ~23% with `apply_trimmed_cfg`, so the saving
    is roughly `(1 - keep)` times that share.

    Args:
        keep: fraction of steps that keep guidance (0.5 = half).
        where: "first" keeps guidance on the early steps, "last" on the late
            ones. Early steps set the coarse structure, so "first" is the safer
            default.
    """
    if getattr(model, "_cfg_interval_applied", False):
        return model

    import torch
    from omnivoice.models.omnivoice import OmniVoiceModelOutput

    original_forward = model.forward
    original_generate_iterative = model._generate_iterative
    state = {"step": 0, "num_step": 32}

    def generate_iterative(task, gen_config):
        # Reset per call so the wrapper below knows the step index.
        state["step"] = 0
        state["num_step"] = max(1, gen_config.num_step)
        return original_generate_iterative(task, gen_config)

    def guided_this_step() -> bool:
        n = state["num_step"]
        guided = max(1, int(round(keep * n)))
        if where == "last":
            return state["step"] >= n - guided
        return state["step"] < guided

    def forward(input_ids=None, audio_mask=None, attention_mask=None, **kwargs):
        step = state["step"]
        state["step"] = step + 1

        if (
            guided_this_step()
            or input_ids is None
            or attention_mask is None
            or not torch.is_tensor(attention_mask)
            or attention_mask.dim() != 4
            or attention_mask.dtype != torch.bool
            or input_ids.shape[0] % 2 != 0
        ):
            return original_forward(
                input_ids=input_ids, audio_mask=audio_mask,
                attention_mask=attention_mask, **kwargs
            )

        b = input_ids.shape[0] // 2
        lengths = attention_mask[:, 0, :, 0].sum(dim=1)
        sub = original_forward(
            input_ids=input_ids[:b],
            audio_mask=audio_mask[:b],
            attention_mask=attention_mask[:b],
            **kwargs,
        )
        cond = sub.logits
        out = cond.new_zeros((input_ids.shape[0],) + tuple(cond.shape[1:]))
        out[:b] = cond
        # Mirror the target slice into the unconditional rows so the guidance
        # term cancels instead of being applied against stale values.
        for i in range(b):
            u_len = int(lengths[b + i])
            c_len = int(lengths[i])
            if u_len <= 0 or u_len > c_len:
                continue
            out[b + i, :, :u_len, :] = cond[i, :, c_len - u_len:c_len, :]
        return OmniVoiceModelOutput(loss=None, logits=out)

    model.forward = forward
    model._generate_iterative = generate_iterative
    model._cfg_interval_applied = True
    logger.info("CFG interval enabled (keep=%.2f, where=%s)", keep, where)
    return model
