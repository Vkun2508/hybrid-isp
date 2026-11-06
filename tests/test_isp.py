"""Automated checks for the ISP core and the API.    pytest -q

The phone-RAW tests use the public sample DNGs in .cache/phone_raw (scripts/test_phone_raw.py documents where
they come from) and are skipped if those files are not present.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ISP_NO_WARMUP", "1")

from isp import io as isp_io  # noqa: E402
from isp import metrics, model_slot, pipelines  # noqa: E402
from isp.classical import denoise, wb  # noqa: E402
from isp.raw import even_crop, pack  # noqa: E402

PHONE_RAW = sorted((ROOT / ".cache" / "phone_raw").glob("*.dng"))
VALUES = {"R": 0.2, "G": 0.5, "B": 0.8}


def _mosaic(pattern: str, h: int = 16, w: int = 16) -> np.ndarray:
    m = np.zeros((h, w), np.float32)
    for i, c in enumerate(pattern):
        m[i // 2::2, i % 2::2] = VALUES[c]
    return m


@pytest.mark.parametrize("pattern", ["RGGB", "GRBG", "GBRG", "BGGR"])
def test_every_bayer_pattern_is_aligned_to_rggb(pattern):
    packed = pack(even_crop(isp_io._to_rggb(_mosaic(pattern), pattern), 2))  # the loader crops to even size too
    assert np.all(packed[..., 0] == VALUES["R"])
    assert np.all(packed[..., 1] == VALUES["G"]) and np.all(packed[..., 2] == VALUES["G"])
    assert np.all(packed[..., 3] == VALUES["B"])


def test_unknown_pattern_is_rejected_with_a_message():
    with pytest.raises(isp_io.InputError) as e:
        isp_io._to_rggb(_mosaic("RGGB"), "RGBW")
    assert e.value.code == "unsupported_cfa"


def test_unprocessing_round_trips_through_the_classical_stages():
    import cv2

    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.random((192, 256, 3)).astype(np.float32), (0, 0), 6) * 2.0 - 0.5
    img = np.clip(img, 0.05, 0.95).astype(np.float32)  # a smooth natural-ish image (no synthetic hard edges)
    raw = isp_io.from_srgb(img)
    out = pipelines.reference_render(raw.bayer, raw)
    assert metrics.psnr(out, raw.gt) > 30.0  # only demosaicing error remains


def test_white_balance_estimate_does_not_change_with_exposure():
    rng = np.random.default_rng(1)
    rgb = rng.random((64, 64, 3)).astype(np.float32) * np.array([0.5, 0.7, 0.4], np.float32)
    rgb[20:24, 20:24] = 0.96  # a specular highlight
    bayer = isp_io.mosaic(rgb)
    a = wb.shades_of_gray(bayer)
    b = wb.shades_of_gray(bayer * 0.25)
    assert wb.angular_error_deg(a, b) < 0.5


def test_noise_model_bound_never_exceeds_k_sigma():
    rng = np.random.default_rng(2)
    noisy = (rng.random((64, 64, 4)) * 0.5).astype(np.float32)
    wild = noisy + rng.normal(0, 0.3, noisy.shape).astype(np.float32)  # a network gone wrong
    a, b = 1e-3, 1e-5
    out = denoise.bound_residual(noisy, wild, 2.0, (a, b))
    sigma = np.sqrt(a * np.maximum(wild, 0) + b)  # sigma is evaluated at the network's estimate
    assert np.all(np.abs(out - noisy) <= 2.0 * sigma + 1e-5)


def test_model_slot_contract_holds_at_odd_sizes():
    m = model_slot.get_model()
    for h, w in ((37, 53), (128, 200)):
        y = m.run(np.random.default_rng(3).random((h, w, 4)).astype(np.float32) * 0.3)
        assert y.shape == (h, w, 4) and np.isfinite(y).all()


@pytest.mark.skipif(not PHONE_RAW, reason="no phone DNGs in .cache/phone_raw")
@pytest.mark.parametrize("path", PHONE_RAW, ids=lambda p: p.stem)
def test_phone_raw_black_and_white_levels(path):
    raw = isp_io.load_any(path.read_bytes(), path.name)
    g = pack(raw.bayer)[..., 1]
    assert abs(float(np.percentile(g, 0.1))) < 0.03  # black level removed: darkest pixels sit at 0
    assert float(g.max()) <= 1.0001  # scaled by the white level
    assert raw.wb_gains is not None and raw.cam2rgb.shape == (3, 3)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from server.main import app

    return TestClient(app)


def test_api_health_and_samples(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert len(client.get("/api/samples").json()) >= 1


def test_api_processes_a_sample(client):
    sid = client.get("/api/samples").json()[0]["id"]
    r = client.post("/api/process", data={"sample_id": sid, "noise": "0.3"})
    assert r.status_code == 200
    body = r.json()
    assert set(body["outputs"]) == {"traditional", "learned", "hybrid"}
    assert client.get(body["outputs"]["hybrid"]["image_url"]).headers["content-type"] == "image/webp"


@pytest.mark.parametrize("payload,status", [
    ({"files": {"file": ("x.dng", b"not a raw file at all" * 20)}}, 422),
    ({"files": {"file": ("notes.txt", b"hello")}}, 415),
    ({"data": {"sample_id": "nope"}}, 404),
    ({"data": {"sample_id": "x", "noise": "5"}}, 422),
    ({}, 400),
])
def test_api_bad_input_gets_a_friendly_error(client, payload, status):
    r = client.post("/api/process", **payload)
    assert r.status_code == status
    err = r.json()["error"]
    assert err["message"] and "code" in err
