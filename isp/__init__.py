"""Hybrid smartphone ISP: classical stages + a pluggable learned core.

Public entry points:
    isp.io.load_any(data, filename)   -> RawImage
    isp.simulate.simulate(raw, ...)   -> RawImage (degraded copy)
    isp.pipelines.run(raw, names)     -> dict of pipeline results
    isp.model_slot.get_model()        -> the loaded learned model (or the stub)
"""
