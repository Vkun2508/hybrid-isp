# The model slot

The app loads the learned core from this folder, described by `model.json`. If it's missing,
or the file it names is missing, a classical stand-in with the same interface runs instead, and the UI
shows "Placeholder model".

To plug in the trained network (from the training notebook's `exported_model.zip`):

1. Copy `isp_unet.onnx`, `isp_unet_int8.onnx` and `model.json` here.
2. Check it: `python scripts/check_model.py` (expect `RESULT : PASS`).
3. Restart the server or redeploy. No code changes are needed.

`model.json` for the team's `LightweightISPUNet` (packed RAW in, denoised packed RAW out):

```json
{ "name": "LightweightISPUNet (SIDD-CC RAW denoiser)", "file": "isp_unet_int8.onnx",
  "input": "raw4", "input_wb": false, "output": "raw4", "upscale": 1,
  "pad_multiple": 8, "layout": "NCHW", "tile": 1024, "overlap": 32 }
```

Every field is documented in `isp/model_slot.py` (`Manifest`).
