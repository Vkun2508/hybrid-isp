"""Convert the team's PyTorch RAW denoiser to ONNX (FP32 + optional static INT8) for the web app.

Same logic as notebook Section 17b, for when you have a .pth and want to (re)export locally:

    python scripts/export_onnx.py --weights models/best_isp_unet.pth --out models/isp_unet.onnx --int8
    python scripts/export_onnx.py --random-init --out models/random_isp_unet.onnx   # integration test only

Needs the dev requirements (torch, onnx). The deploy image only needs onnxruntime.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_class(model_py: Path, cls: str):
    spec = importlib.util.spec_from_file_location("user_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return getattr(mod, cls)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-py", default=str(ROOT / "training" / "unet_isp.py"))
    ap.add_argument("--cls", default="LightweightISPUNet")
    ap.add_argument("--kwargs", default='{"in_channels": 4, "out_channels": 4, "base_channels": 32}')
    ap.add_argument("--weights", help="state_dict or training checkpoint (.pth)")
    ap.add_argument("--random-init", action="store_true", help="export untrained weights (integration test only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--int8", action="store_true", help="also write <out>_int8.onnx (static QDQ, calibrated)")
    ap.add_argument("--calib", help="optional .npz with key 'packed' (N, 4, h, w) calibration patches")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    Net = load_class(Path(args.model_py), args.cls)
    net = Net(**json.loads(args.kwargs)).eval()
    if args.weights:
        ck = torch.load(args.weights, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck)
    elif not args.random_init:
        ap.error("give --weights, or --random-init for an integration test")

    class NoPad(nn.Module):
        """forward() without the internal pad/crop; the app pads to a multiple of 8 itself."""
        def __init__(self, n):
            super().__init__()
            self.net = n

        def forward(self, x):
            n = self.net
            e1 = n.enc1(n.head(x)); d1 = n.down1(e1)
            e2 = n.enc2(d1); d2 = n.down2(e2)
            e3 = n.enc3(d2); d3 = n.down3(e3)
            b = n.bottleneck(d3)
            dec3 = n.dec3(torch.cat([n.up3(b), e3], 1))
            dec2 = n.dec2(torch.cat([n.up2(dec3), e2], 1))
            dec1 = n.dec1(torch.cat([n.up1(dec2), e1], 1))
            return x + n.tail(dec1)

    wrapped = NoPad(net).eval()
    with torch.no_grad():
        xt = torch.rand(1, 4, 64, 96)
        assert torch.allclose(net(xt), wrapped(xt), atol=1e-6), "wrapper differs from model.forward()"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 4, 256, 256)
    names = dict(input_names=["packed_raw"], output_names=["denoised_raw"])
    axes = {"packed_raw": {2: "h", 3: "w"}, "denoised_raw": {2: "h", 3: "w"}}
    try:
        torch.onnx.export(wrapped, (dummy,), str(out), opset_version=17, dynamic_axes=axes, dynamo=False, **names)
        exporter = "torchscript"
    except Exception as e:
        print("TorchScript exporter failed:", type(e).__name__, e, "-> trying dynamo")
        from torch.export import Dim
        h8, w8 = Dim("h8", min=1, max=2048), Dim("w8", min=1, max=2048)
        prog = torch.onnx.export(wrapped, (dummy,), dynamo=True, dynamic_shapes={"x": {2: 8 * h8, 3: 8 * w8}}, **names)
        prog.save(str(out))
        exporter = "dynamo"

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(out))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    parity = {}
    for hw in [(256, 256), (192, 320)]:
        x = torch.rand(1, 4, *hw) * 0.5
        with torch.no_grad():
            ref = wrapped(x).numpy()
        parity[f"{hw[0]}x{hw[1]}"] = float(np.abs(ref - sess.run(None, {"packed_raw": x.numpy()})[0]).max())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, exporter={exporter}); parity {parity}")
    if max(parity.values()) > 1e-4:
        print("PARITY FAILED")
        return 1

    if args.int8:
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

        if args.calib:
            arrs = [a[None].astype(np.float32) for a in np.load(args.calib)["packed"][:32]]
        else:  # synthetic: smooth signal + Poisson-Gaussian-like noise in RAW range
            rng = np.random.default_rng(0)
            arrs = []
            for _ in range(16):
                base = rng.random((1, 4, 16, 16)).astype(np.float32) * 0.4
                big = np.kron(base, np.ones((1, 1, 8, 8), np.float32))
                arrs.append(big + rng.normal(0, 0.02, big.shape).astype(np.float32))

        class Calib(CalibrationDataReader):
            def __init__(self, a):
                self.it = iter([{"packed_raw": x} for x in a])

            def get_next(self):
                return next(self.it, None)

        g = onnx.load(str(out)).graph
        keep = [n.name for n in g.node if n.op_type == "Add" and "denoised_raw" in n.output]
        tail = [n.name for n in g.node if n.op_type == "Conv"][-1:]
        q = out.with_name(out.stem + "_int8.onnx")
        quantize_static(str(out), str(q), Calib(arrs), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, op_types_to_quantize=["Conv"],
                        nodes_to_exclude=keep + tail, extra_options={"DedicatedQDQPair": True})
        s8 = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"])
        x1 = (np.random.rand(1, 4, 512, 512).astype(np.float32) * 0.3)
        for name, s in (("fp32", sess), ("int8", s8)):
            s.run(None, {"packed_raw": x1})
            t0 = time.perf_counter()
            for _ in range(5):
                s.run(None, {"packed_raw": x1})
            print(f"  {name}: {(time.perf_counter() - t0) / 5 * 1000:.0f} ms at 1 MP RAW (this CPU)")
        print(f"wrote {q} ({q.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
