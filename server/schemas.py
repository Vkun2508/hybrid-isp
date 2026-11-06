"""API response shapes (the contract the web UI is built against). Served at /docs too."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    hint: str = ""


class ErrorResponse(BaseModel):
    error: ErrorBody


class Metrics(BaseModel):
    psnr: float = Field(description="dB, higher is better")
    ssim: float = Field(description="0..1, higher is better")
    delta_e: float = Field(description="mean CIEDE2000, lower is better")


class ModelInfo(BaseModel):
    name: str
    kind: str
    is_stub: bool
    file: str | None = None
    input: str
    output: str
    params: int | None = None
    size_mb: float | None = None
    load_error: str | None = None


class Health(BaseModel):
    status: str
    version: str
    model: ModelInfo
    max_megapixels: float
    samples: int


class Sample(BaseModel):
    id: str
    name: str
    description: str
    kind: str
    has_gt: bool
    preset: dict
    source: dict
    thumb_url: str


class StageOut(BaseModel):
    name: str
    ms: float
    note: str
    image_url: str


class PipelineOut(BaseModel):
    image_url: str
    latency_ms: float
    metrics: Metrics | None = None
    stages: list[StageOut]
    info: dict = {}


class InputInfo(BaseModel):
    name: str
    kind: str
    source: str
    width: int
    height: int
    orig_width: int
    orig_height: int
    pattern: str
    exposure: float
    has_gt: bool
    notes: list[str]
    image_url: str
    gt_url: str | None = None
    metrics: Metrics | None = Field(None, description="the unprocessed input vs ground truth (the baseline to beat)")


class Analysis(BaseModel):
    sigma: float
    ev: float
    clipped: float
    wb_conf: float
    wb_angle: float | None
    dynamic_range: float
    mean_level: float


class Route(BaseModel):
    path: str
    label: str
    gain: float
    wb_method: str
    wb_gains: list[float]
    local_tone: bool
    sharpen: float
    chroma: float
    sigma_eff: float
    forced: bool
    reasons: list[str]


class Simulation(BaseModel):
    noise: float
    exposure_ev: float
    wb_shift: float
    seed: int
    iso: int


class ProcessResponse(BaseModel):
    id: str
    cached: bool
    input: InputInfo
    simulate: Simulation
    analysis: Analysis
    route: Route
    model: ModelInfo
    outputs: dict[str, PipelineOut]
    timings: dict[str, float]
