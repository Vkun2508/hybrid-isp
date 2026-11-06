import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ApiError, ProcessResponse } from "../api";
import { fixed } from "../format";

type Key = "input" | "traditional" | "learned" | "hybrid" | "gt";
const LABEL: Record<Key, string> = {
  input: "Input (sensor, no ISP)", traditional: "Traditional", learned: "Learned only", hybrid: "Hybrid", gt: "Ground truth",
};
const ZOOMS = [1, 2, 4] as const;

function urlOf(r: ProcessResponse, k: Key): string | null {
  if (k === "input") return r.input.image_url;
  if (k === "gt") return r.input.gt_url;
  return r.outputs[k]?.image_url ?? null;
}

interface Props {
  result: ProcessResponse | null;
  busy: boolean;
  elapsed: number;
  error: ApiError | null;
  onRetry: () => void;
}

export function CompareViewer({ result, busy, elapsed, error, onRetry }: Props) {
  const [left, setLeft] = useState<Key>("traditional");
  const [right, setRight] = useState<Key>("hybrid");
  const [split, setSplit] = useState(50);
  const [lens, setLens] = useState({ x: 0.3, y: 0.42 });
  const [zoom, setZoom] = useState<(typeof ZOOMS)[number]>(2);
  const [tile, setTile] = useState({ w: 240, h: 160 });
  const [boxW, setBoxW] = useState(800);
  const areaRef = useRef<HTMLDivElement>(null);
  const tileRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const hasGt = !!result?.input.gt_url;
  const keys: Key[] = ["input", "traditional", "learned", "hybrid", ...(hasGt ? (["gt"] as Key[]) : [])];
  useEffect(() => {
    if (!hasGt && left === "gt") setLeft("traditional");
    if (!hasGt && right === "gt") setRight("hybrid");
  }, [hasGt, left, right]);

  // measure the loupe tiles and the image box (they set the lens maths)
  useLayoutEffect(() => {
    const ro = new ResizeObserver(() => {
      if (tileRef.current) setTile({ w: tileRef.current.clientWidth, h: tileRef.current.clientHeight });
      if (areaRef.current) setBoxW(areaRef.current.clientWidth);
    });
    if (tileRef.current) ro.observe(tileRef.current);
    if (areaRef.current) ro.observe(areaRef.current);
    return () => ro.disconnect();
  }, [result === null]);

  const W = result?.input.width ?? 3;
  const H = result?.input.height ?? 2;

  const pointTo = (e: React.PointerEvent) => {
    const r = areaRef.current?.getBoundingClientRect();
    if (!r) return null;
    return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) };
  };

  const onAreaMove = (e: React.PointerEvent) => {
    if (dragging.current) {
      const p = pointTo(e);
      if (p) setSplit(p.x * 100);
      return;
    }
    if (e.pointerType === "mouse" || e.buttons) {
      const p = pointTo(e);
      if (p) setLens(p);
    }
  };

  // the image region one loupe tile covers, drawn as a box on the viewer
  const boxW_ = (tile.w / zoom / W) * boxW;
  const boxH_ = (tile.h / zoom / W) * boxW;
  const lensStyle = (k: Key) => {
    const url = result ? urlOf(result, k) : null;
    const cx = lens.x * W, cy = lens.y * H;
    return url
      ? {
          backgroundImage: `url(${url})`,
          backgroundSize: `${W * zoom}px ${H * zoom}px`,
          backgroundPosition: `${tile.w / 2 - cx * zoom}px ${tile.h / 2 - cy * zoom}px`,
        }
      : {};
  };
  const lensKeys: Key[] = ["traditional", "learned", "hybrid", ...(hasGt ? (["gt"] as Key[]) : [])];

  return (
    <section className="card o-viewer" aria-labelledby="viewer-h">
      <div className="viewer-head">
        <h2 id="viewer-h" className="sr-only">Compare pipelines</h2>
        <label className="sr-only" htmlFor="left-sel">Left image</label>
        <select id="left-sel" className="select" value={left} onChange={(e) => setLeft(e.target.value as Key)}>
          {keys.map((k) => <option key={k} value={k}>{LABEL[k]}</option>)}
        </select>
        <button className="btn icon" aria-label="Swap left and right" title="Swap" onClick={() => { setLeft(right); setRight(left); }}>
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M2 5h11M10 2l3 3-3 3M14 11H3M6 8l-3 3 3 3" fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>
        </button>
        <label className="sr-only" htmlFor="right-sel">Right image</label>
        <select id="right-sel" className="select" value={right} onChange={(e) => setRight(e.target.value as Key)}>
          {keys.map((k) => <option key={k} value={k}>{LABEL[k]}</option>)}
        </select>
        <span className="spacer" />
        <span className="small muted num">{result ? `${result.input.width} × ${result.input.height}` : ""}</span>
      </div>

      <div className="well">
        {busy && <div className="progress" role="progressbar" aria-label="Processing" />}
        <div
          ref={areaRef}
          className={"compare" + (busy ? " busy" : "")}
          style={{ aspectRatio: `${W} / ${H}`, width: `min(100%, calc(50vh * ${W / H}))` }}
          onPointerMove={onAreaMove}
          onPointerDown={(e) => { const p = pointTo(e); if (p && !dragging.current) setLens(p); }}
        >
          {result ? (
            <>
              <img src={urlOf(result, left) ?? ""} alt={`${LABEL[left]} result`} draggable={false} />
              <img className="layer-b" src={urlOf(result, right) ?? ""} alt={`${LABEL[right]} result`} draggable={false}
                style={{ ["--split" as string]: `${split}%` }} />
              <span className="lens-box" style={{
                width: boxW_, height: boxH_,
                left: `calc(${lens.x * 100}% - ${boxW_ / 2}px)`, top: `calc(${lens.y * 100}% - ${boxH_ / 2}px)`,
              }} />
              <span className="split-line" style={{ ["--split" as string]: `${split}%` }} />
              <button
                className="split-handle" style={{ ["--split" as string]: `${split}%` }}
                role="slider" aria-label="Comparison divider" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(split)}
                aria-valuetext={`${Math.round(split)}% ${LABEL[left]}`}
                onPointerDown={(e) => { e.stopPropagation(); dragging.current = true; (e.target as HTMLElement).setPointerCapture(e.pointerId); }}
                onPointerMove={(e) => { if (dragging.current) { const p = pointTo(e); if (p) setSplit(p.x * 100); } }}
                onPointerUp={() => { dragging.current = false; }}
                onPointerCancel={() => { dragging.current = false; }}
                onKeyDown={(e) => {
                  const step = e.shiftKey ? 10 : 2;
                  if (e.key === "ArrowLeft") setSplit((s) => Math.max(0, s - step));
                  else if (e.key === "ArrowRight") setSplit((s) => Math.min(100, s + step));
                  else if (e.key === "Home") setSplit(0);
                  else if (e.key === "End") setSplit(100);
                  else return;
                  e.preventDefault();
                }}
              >
                <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path d="M7 4L2 9l5 5M11 4l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="1.6" /></svg>
              </button>
              <span className="corner left">{LABEL[left]}</span>
              <span className={"corner right" + (right === "hybrid" ? " hybrid" : "")}>{LABEL[right]}</span>
            </>
          ) : (
            <div className="skel" style={{ position: "absolute", inset: 0 }} aria-label="Loading image" />
          )}
        </div>
        {busy && <span className="pill status-chip num" role="status">Processing on CPU · {(elapsed / 1000).toFixed(1)} s</span>}
        {error && !busy && (
          <div className="viewer-error" role="alert">
            <div className="box">
              <span className="err-mark" />
              <strong>{error.message}</strong>
              {error.hint && <span className="secondary">{error.hint}</span>}
              <button className="btn" onClick={onRetry} style={{ marginTop: 8 }}>Try again</button>
            </div>
          </div>
        )}
      </div>

      <div className="loupe-head">
        <h3>Pixel loupe <span className="muted small" style={{ fontWeight: 400 }}>· the same spot in every pipeline · move over the photo</span></h3>
        <div className="seg" role="group" aria-label="Loupe zoom">
          {ZOOMS.map((z) => (
            <button key={z} aria-pressed={zoom === z} onClick={() => setZoom(z)} className="num">{z === 1 ? "1:1" : `${z}×`}</button>
          ))}
        </div>
      </div>
      <div className="loupe" style={{ ["--n" as string]: lensKeys.length }}>
        {lensKeys.map((k, i) => {
          const m = k === "gt" || k === "input" ? null : result?.outputs[k]?.metrics;
          return (
            <figure key={k} className={k === "hybrid" ? "hybrid" : undefined}>
              <div ref={i === 0 ? tileRef : undefined} className={"lens" + (result ? "" : " skel")} style={lensStyle(k)}
                role="img" aria-label={`${LABEL[k]}, magnified`} />
              <figcaption>
                <b>{k === "gt" ? "Ground truth" : LABEL[k]}</b>
                {m && <span className="num">{fixed(m.psnr, 1)} dB</span>}
              </figcaption>
            </figure>
          );
        })}
      </div>
    </section>
  );
}
