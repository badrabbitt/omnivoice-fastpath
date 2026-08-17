"""Build the ONNX CPU backend for OmniVoice — portable across macOS/Linux/Windows.

Captures a real forward input, exports the hot path to ONNX, then makes an int8
version that keeps the audio-head projection in float32 (quantizing that one node
is what wrecks quality — argmax agreement drops from 77% to 51%).

    pip install onnxruntime onnx onnxscript
    python make_onnx.py --out-dir ./onnx

Produces <out-dir>/omnivoice_fp32.onnx (+ .data) and <out-dir>/omnivoice_int8.onnx.
Point the service at the int8 file with DUBBING_OMNIVOICE_ONNX.
"""
import argparse, collections, os, sys, time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="./onnx")
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-fp32", action="store_true",
                    help="reuse an existing fp32 export and only re-quantize")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fp32_path = os.path.join(out_dir, "omnivoice_fp32.onnx")
    int8_path = os.path.join(out_dir, "omnivoice_int8.onnx")

    torch.set_num_threads(args.threads)

    if not args.skip_fp32:
        from omnivoice.models.omnivoice import OmniVoice

        print(f"Loading {args.model} on CPU ...", flush=True)
        model = OmniVoice.from_pretrained(args.model, device_map="cpu",
                                          dtype=torch.float32).eval()

        # Capture one real set of forward inputs; shapes are made dynamic below,
        # so any single sample is enough to trace the graph.
        captured = {}
        inner = model.forward

        def spy(input_ids=None, audio_mask=None, attention_mask=None, **kw):
            if "input_ids" not in captured:
                captured.update(input_ids=input_ids.clone(),
                                audio_mask=audio_mask.clone(),
                                attention_mask=attention_mask.clone())
            return inner(input_ids=input_ids, audio_mask=audio_mask,
                         attention_mask=attention_mask, **kw)

        model.forward = spy
        model.generate(text="Xin chào các bạn, đây là một câu để lấy mẫu đầu vào.",
                       language="vi", num_step=2)
        model.forward = inner

        # generate() runs under torch.inference_mode(), and inference tensors
        # cannot be traced by torch.export. Round-trip through numpy to get
        # ordinary tensors back.
        ids, amask, attn = (
            torch.from_numpy(captured[k].cpu().numpy())
            for k in ("input_ids", "audio_mask", "attention_mask")
        )
        print(f"sample shapes: {tuple(ids.shape)} {tuple(amask.shape)} {tuple(attn.shape)}",
              flush=True)

        class Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, audio_mask, attention_mask):
                return self.m(input_ids=input_ids, audio_mask=audio_mask,
                              attention_mask=attention_mask).logits

        wrapper = Wrapper(model).eval()
        batch = torch.export.Dim("batch", min=1, max=64)
        seq = torch.export.Dim("seq", min=8, max=4096)

        t0 = time.time()
        torch.onnx.export(
            wrapper, (ids, amask, attn), fp32_path,
            input_names=["input_ids", "audio_mask", "attention_mask"],
            output_names=["logits"],
            dynamic_shapes={"input_ids": {0: batch, 2: seq},
                            "audio_mask": {0: batch, 1: seq},
                            "attention_mask": {0: batch, 2: seq, 3: seq}},
            dynamo=True, external_data=True, opset_version=18,
        )
        print(f"exported fp32 in {time.time() - t0:.0f}s -> {fp32_path}", flush=True)
        del model, wrapper

    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    graph = onnx.load(fp32_path, load_external_data=False)
    init_shapes = {i.name: tuple(i.dims) for i in graph.graph.initializer}
    exclude, shapes = [], collections.Counter()
    for node in graph.graph.node:
        if node.op_type != "MatMul":
            continue
        shape = next((init_shapes[i] for i in node.input if i in init_shapes), None)
        if shape is None:
            continue  # activation x activation; MatMulConstBOnly skips these
        shapes[shape] += 1
        # The audio head maps hidden -> num_codebooks * vocab. Quantizing it puts
        # noise straight into token selection.
        if max(shape) > 4096:
            exclude.append(node.name)

    print("MatMul weight shapes:", dict(shapes), flush=True)
    print(f"keeping {len(exclude)} node(s) in float32", flush=True)

    t0 = time.time()
    quantize_dynamic(model_input=fp32_path, model_output=int8_path,
                     weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                     per_channel=True, reduce_range=False,
                     nodes_to_exclude=exclude,
                     extra_options={"MatMulConstBOnly": True})
    print(f"quantized in {time.time() - t0:.0f}s -> {int8_path}", flush=True)
    print(f"\nDUBBING_OMNIVOICE_ONNX={int8_path}")


if __name__ == "__main__":
    main()
