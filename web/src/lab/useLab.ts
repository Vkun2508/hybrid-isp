import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError, MAX_UPLOAD_MB, getHealth, getSamples, processImage,
  type ModelInfo, type ProcessResponse, type RouteMode, type Sample, type SimPreset,
} from "../api";
import type { ImageKey } from "../format";

export type Source = { kind: "sample"; sample: Sample } | { kind: "file"; file: File };
export type Pair = [ImageKey, ImageKey];
export type Zoom = 1 | 2 | 4;

/** Fetch the output images before showing a result, so numbers and pictures change together (capped, never hangs). */
function preload(urls: string[], capMs = 8000): Promise<void> {
  const all = Promise.all(urls.map((u) => new Promise<void>((res) => {
    const im = new Image();
    im.onload = () => res();
    im.onerror = () => res();
    im.src = u;
  })));
  return Promise.race([all.then(() => undefined), new Promise<void>((res) => setTimeout(res, capMs))]);
}

export const ZERO: SimPreset = { noise: 0, exposure_ev: 0, wb_shift: 0 };
export const sameSim = (a: SimPreset, b: SimPreset) =>
  a.noise === b.noise && a.exposure_ev === b.exposure_ev && a.wb_shift === b.wb_shift;

/** All Lab state in one place, so the Lab screen, the Overview and the guided tour can share it. */
export function useLab() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [samples, setSamples] = useState<Sample[] | null>(null);
  const [source, setSource] = useState<Source | null>(null);
  const [sim, setSim] = useState<SimPreset>(ZERO);
  const [route, setRoute] = useState<RouteMode>("auto");
  const [result, setResult] = useState<ProcessResponse | null>(null);
  const [resultSource, setResultSource] = useState<Source | null>(null); // which input the shown result belongs to
  const [busy, setBusy] = useState(false);
  const [loadingImages, setLoadingImages] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [startedAt, setStartedAt] = useState(0);
  const [now, setNow] = useState(0);
  // viewer state (lifted so the tour can drive it)
  const [pair, setPair] = useState<Pair>(["traditional", "hybrid"]);
  const [split, setSplit] = useState(50);
  const [lens, setLens] = useState({ x: 0.62, y: 0.62 });
  const [zoom, setZoom] = useState<Zoom>(2);
  const ctrl = useRef<AbortController | null>(null);

  // boot: model status + samples; the first sample is precomputed on the server, so it lands instantly
  useEffect(() => {
    getHealth().then((h) => setModel(h.model)).catch(() => undefined);
    getSamples()
      .then((s) => {
        setSamples(s);
        if (s.length) {
          setSource((cur) => cur ?? { kind: "sample", sample: s[0] });
          setSim((cur) => (sameSim(cur, ZERO) ? s[0].preset : cur));
        }
      })
      .catch((e) => setError(e as ApiError));
  }, []);

  const run = useCallback((src: Source, s: SimPreset, rt: RouteMode) => {
    ctrl.current?.abort();
    const c = new AbortController();
    ctrl.current = c;
    setBusy(true);
    setLoadingImages(false);
    setError(null);
    setStartedAt(performance.now());
    processImage({
      sampleId: src.kind === "sample" ? src.sample.id : undefined,
      file: src.kind === "file" ? src.file : undefined,
      sim: s, route: rt, signal: c.signal,
    })
      .then(async (r) => {
        if (c.signal.aborted) return;
        setLoadingImages(true);
        await preload([r.outputs.traditional?.image_url, r.outputs.hybrid?.image_url, r.outputs.learned?.image_url, r.input.gt_url]
          .filter((u): u is string => !!u));
        if (c.signal.aborted) return;
        setResult(r);
        setResultSource(src);
        setModel(r.model);
      })
      .catch((e) => {
        if ((e as Error).name === "AbortError") return;
        setError(e instanceof ApiError ? e : new ApiError("unknown", "Something went wrong.", "Try again.", 0));
      })
      .finally(() => {
        if (ctrl.current === c) { setBusy(false); setLoadingImages(false); }
      });
  }, []);

  // any change of input, simulator or router reprocesses (debounced while a slider is being dragged)
  useEffect(() => {
    if (!source) return;
    const t = setTimeout(() => run(source, sim, route), 350);
    return () => clearTimeout(t);
  }, [source, sim, route, run]);

  // elapsed counter while processing
  useEffect(() => {
    if (!busy) return;
    const id = setInterval(() => setNow(performance.now()), 100);
    return () => clearInterval(id);
  }, [busy]);

  // no ground truth for this photo: never leave the viewer pointing at it
  const hasGt = !!result?.input.gt_url;
  useEffect(() => {
    if (!hasGt && (pair[0] === "gt" || pair[1] === "gt")) setPair(["traditional", "hybrid"]);
  }, [hasGt, pair]);

  const pickSample = useCallback((s: Sample, preset?: SimPreset) => {
    setSource({ kind: "sample", sample: s });
    setSim(preset ?? s.preset);
  }, []);
  const pickFile = useCallback((f: File) => {
    if (f.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setError(new ApiError("too_large", `That file is over ${MAX_UPLOAD_MB} MB.`, "Use a smaller photo, or export a JPG from your phone.", 413));
      return;
    }
    setSource({ kind: "file", file: f });
    setSim(ZERO);
  }, []);
  const retry = useCallback(() => { if (source) run(source, sim, route); }, [source, sim, route, run]);
  const backToSamples = useCallback(() => { if (samples?.length) { setSource({ kind: "sample", sample: samples[0] }); setSim(samples[0].preset); } }, [samples]);

  const preset = source?.kind === "sample" ? source.sample.preset : ZERO;
  return {
    model, samples, source, sim, route, result, busy, loadingImages, error, preset,
    elapsed: busy ? Math.max(0, now - startedAt) : 0,
    /** the result on screen is for an older input (still processing, or the new one failed) */
    stale: !!result && (busy || resultSource !== source),
    pair, split, lens, zoom,
    setSim, setRoute, setPair, setSplit, setLens, setZoom, pickSample, pickFile, retry, backToSamples,
  };
}

export type Lab = ReturnType<typeof useLab>;
