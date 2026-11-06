// Typed client for the FastAPI server (mirrors server/schemas.py).

export type PipelineName = "traditional" | "learned" | "hybrid";
export const PIPELINES: PipelineName[] = ["traditional", "learned", "hybrid"];

export interface Metrics { psnr: number; ssim: number; delta_e: number }

export interface ModelInfo {
  name: string; kind: string; is_stub: boolean; file: string | null; input: string; output: string;
  params: number | null; size_mb: number | null; load_error: string | null;
}
export interface Health { status: string; version: string; model: ModelInfo; max_megapixels: number; samples: number }

export interface SimPreset { noise: number; exposure_ev: number; wb_shift: number }
export interface Sample {
  id: string; name: string; description: string; kind: "sidd" | "image"; has_gt: boolean;
  preset: SimPreset; source: Record<string, string | number>; thumb_url: string;
}

export interface StageOut { name: string; ms: number; note: string; image_url: string }
export interface PipelineOut {
  image_url: string; latency_ms: number; metrics: Metrics | null; stages: StageOut[]; info: Record<string, unknown>;
}
export interface InputInfo {
  name: string; kind: string; source: string; width: number; height: number; orig_width: number; orig_height: number;
  pattern: string; exposure: number; has_gt: boolean; notes: string[]; image_url: string; gt_url: string | null;
  metrics: Metrics | null;
}
export interface Analysis {
  sigma: number; ev: number; clipped: number; wb_conf: number; wb_angle: number | null; dynamic_range: number; mean_level: number;
}
export interface Route {
  path: "fast" | "learned" | "learned_chroma"; label: string; gain: number; wb_method: string; wb_gains: number[];
  local_tone: boolean; sharpen: number; chroma: number; sigma_eff: number; forced: boolean; reasons: string[];
}
export interface ProcessResponse {
  id: string; cached: boolean; input: InputInfo;
  simulate: SimPreset & { seed: number; iso: number };
  analysis: Analysis; route: Route; model: ModelInfo;
  outputs: Record<PipelineName, PipelineOut>;
  timings: Record<string, number>;
}

type Cell = { psnr: number; delta_e: number };

// results/benchmarks.json, written by eval/*.py (the Results tab renders whatever subset exists)
export interface Benchmarks {
  status: "final" | "provisional" | "pending";
  message?: string;
  updated?: string;
  model?: { name: string; params: number; variant: string };
  dataset?: { name: string; test_scenes: string[]; n_images: number; note?: string };
  quality?: { pipeline: string; psnr: number; ssim: number; delta_e: number; lpips?: number | null }[];
  efficiency?: { variant: string; params: number; size_mb: number; latency_ms_1mp: number; peak_mem_mb?: number | null; raw_psnr_db?: number | null }[];
  robustness?: { title: string; x_label: string; x: number[]; series: { pipeline: string; psnr: number[] }[] }[];
  ablation?: { variant: string; psnr: number; ssim: number; delta_e: number; note?: string }[];
  failures?: { title: string; description: string; image_url?: string }[];
  meta?: { subset_n: number; ablation_conditions: string[]; robustness_note: string };
  router?: {
    conditions: string[]; n_images: number;
    modes: { mode: string; psnr: number; delta_e: number; latency_ms: number; network_share: number }[];
    paths_by_iso_as_captured: Record<string, Record<string, number>>;
  };
  breakdown?: Record<"by_phone" | "by_iso", { group: string; n: number; input: Cell; traditional: Cell; learned: Cell; hybrid: Cell }[]>;
  efficiency_sweep?: Record<string, number | string>[];
  tiling?: { image: string; note?: string; variants: { variant: string; latency_ms: number; psnr_vs_single_pass: number; interior_max_abs_diff?: number; interior_psnr?: number }[] };
  phone_raw?: {
    file: string; preview?: string; passed: boolean;
    libraw?: { raw_type: string; cfa: string; black_level_per_channel: number[]; white_level: number };
    route?: { path: string; wb: string };
    checks?: { colour_vs_camera_wb_deg: number };
  }[];
}

export class ApiError extends Error {
  constructor(public code: string, message: string, public hint: string, public status: number) {
    super(message);
  }
}

async function parse<T>(res: Response): Promise<T> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON (proxy error page, network cut) */
  }
  if (!res.ok) {
    const err = (body as { error?: { code: string; message: string; hint: string } } | null)?.error;
    if (err) throw new ApiError(err.code, err.message, err.hint, res.status);
    throw new ApiError("http_" + res.status, "The server didn't answer properly.", "Try again in a moment.", res.status);
  }
  return body as T;
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("offline", "Couldn't reach the server.", "Check your connection, then try again.", 0);
  }
  return parse<T>(res);
}

export const getHealth = () => call<Health>("/api/health");
export const getSamples = () => call<Sample[]>("/api/samples");
export const getBenchmarks = () => call<Benchmarks>("/api/benchmarks");

export type RouteMode = "auto" | "learned" | "fast";

export interface ProcessRequest {
  sampleId?: string;
  file?: File;
  sim: SimPreset;
  route: RouteMode;
  signal?: AbortSignal;
}

export function processImage(req: ProcessRequest): Promise<ProcessResponse> {
  const fd = new FormData();
  if (req.file) fd.append("file", req.file, req.file.name);
  else if (req.sampleId) fd.append("sample_id", req.sampleId);
  fd.append("noise", String(req.sim.noise));
  fd.append("exposure_ev", String(req.sim.exposure_ev));
  fd.append("wb_shift", String(req.sim.wb_shift));
  fd.append("force_path", req.route === "auto" ? "auto" : req.route);
  return call<ProcessResponse>("/api/process", { method: "POST", body: fd, signal: req.signal });
}

export const MAX_UPLOAD_MB = 60;
export const ACCEPT =
  ".dng,.cr2,.cr3,.nef,.nrw,.arw,.orf,.rw2,.raf,.pef,.srw,.jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp";
