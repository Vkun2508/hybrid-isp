import { useLayoutEffect, useRef, useState } from "react";
import type { ApiError, ProcessResponse } from "../api";
import { LABEL, SHORT, cls, fixed, type ImageKey } from "../format";
import type { Pair, Zoom } from "../lab/useLab";

function urlOf(r: ProcessResponse, k: ImageKey): string | null {
  if (k === "input") return r.input.image_url;
  if (k === "gt") return r.input.gt_url;
  return r.outputs[k]?.image_url ?? null;
}

const QUICK: { pair: Pair; needsGt?: boolean }[] = [
  { pair: ["traditional", "hybrid"] },
  { pair: ["learned", "hybrid"] },
  { pair: ["input", "hybrid"] },
  { pair: ["hybrid", "gt"], needsGt: true },
];
const KEYS: ImageKey[] = ["input", "traditional", "learned", "hybrid", "gt"];

interface Props {
  result: ProcessResponse | null;
  busy: boolean;
  elapsed: number;
  error: ApiError | null;
  onRetry: () => void;
  pair: Pair;
  onPair: (p: Pair) => void;
  split: number;
  onSplit: (v: number) => void;
  lens: { x: number; y: number };
  onLens: (p: { x: number; y: number }) => void;
  /** the loupe tile size in px, to draw the region it covers */
  tile: { w: number; h: number };
  zoom: Zoom;
}

export function CompareViewer({ result, busy, elapsed, error, onRetry, pair, onPair, split, onSplit, lens, onLens, tile, zoom }: Props) {
  const [custom, setCustom] = useState(false);
  const [boxW, setBoxW] = useState(800);
  const areaRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const [left, right] = pair;
  const hasGt = !!result?.input.gt_url;
  const keys = KEYS.filter((k) => k !== "gt" || hasGt);
  const quick = QUICK.filter((q) => !q.needsGt || hasGt);
  const isQuick = (p: Pair) => p[0] === left && p[1] === right;
  const onQuick = quick.some((q) => isQuick(q.pair));

  useLayoutEffect(() => {
    const ro = new ResizeObserver(() => { if (areaRef.current) setBoxW(areaRef.current.clientWidth); });
    if (areaRef.current) ro.observe(areaRef.current);
    return () => ro.disconnect();
  }, []);

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
      if (p) onSplit(p.x * 100);
      return;
    }
    if (e.pointerType === "mouse" || e.buttons) {
      const p = pointTo(e);
      if (p) onLens(p);
    }
  };

  // the image region one loupe tile covers, drawn as a box on the viewer
  const lbW = (tile.w / zoom / W) * boxW;
  const lbH = (tile.h / zoom / W) * boxW;
  const leftM = left !== "input" && left !== "gt" ? result?.outputs[left]?.metrics : null;
  const rightM = right !== "input" && right !== "gt" ? result?.outputs[right]?.metrics : null;

  return (
    <section className="viewer" id="viewer" aria-labelledby="viewer-h">
      <h2 id="viewer-h" className="sr-only">Compare pipelines</h2>
      <div className="well">
        {busy && <div className="progress" role="progressbar" aria-label="Processing" />}
        <div
          ref={areaRef}
          className={"compare" + (busy ? " busy" : "")}
          style={{ aspectRatio: `${W} / ${H}`, ["--ar" as string]: W / H }}
          onPointerMove={onAreaMove}
          onPointerDown={(e) => { const p = pointTo(e); if (p && !dragging.current) onLens(p); }}
        >
          {result ? (
            <>
              <img src={urlOf(result, left) ?? ""} alt={`${LABEL[left]} result`} draggable={false} />
              <img className="layer-b" src={urlOf(result, right) ?? ""} alt={`${LABEL[right]} result`} draggable={false}
                style={{ ["--split" as string]: `${split}%` }} />
              <span className="lens-box" style={{
                width: lbW, height: lbH,
                left: `calc(${lens.x * 100}% - ${lbW / 2}px)`, top: `calc(${lens.y * 100}% - ${lbH / 2}px)`,
              }} />
              <span className="split-line" style={{ ["--split" as string]: `${split}%` }} />
              <button
                className="split-handle" style={{ ["--split" as string]: `${split}%` }}
                role="slider" aria-label="Comparison divider" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(split)}
                aria-valuetext={`${Math.round(split)}% ${LABEL[left]}`}
                onPointerDown={(e) => { e.stopPropagation(); dragging.current = true; (e.target as HTMLElement).setPointerCapture(e.pointerId); }}
                onPointerMove={(e) => { if (dragging.current) { const p = pointTo(e); if (p) onSplit(p.x * 100); } }}
                onPointerUp={() => { dragging.current = false; }}
                onPointerCancel={() => { dragging.current = false; }}
                onKeyDown={(e) => {
                  const step = e.shiftKey ? 10 : 2;
                  if (e.key === "ArrowLeft") onSplit(Math.max(0, split - step));
                  else if (e.key === "ArrowRight") onSplit(Math.min(100, split + step));
                  else if (e.key === "Home") onSplit(0);
                  else if (e.key === "End") onSplit(100);
                  else return;
                  e.preventDefault();
                }}
              >
                <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path d="M7 4L2 9l5 5M11 4l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="1.8" /></svg>
              </button>
              <span className={"corner left " + cls(left)}>
                <i className="sw" />{SHORT[left]}{leftM && <span className="num">{fixed(leftM.psnr, 1)} dB</span>}
              </span>
              <span className={"corner right " + cls(right)}>
                <i className="sw" />{SHORT[right]}{rightM && <span className="num">{fixed(rightM.psnr, 1)} dB</span>}
              </span>
            </>
          ) : (
            <div className="skel" style={{ position: "absolute", inset: 0 }} aria-label="Loading image" />
          )}
        </div>
        {busy && (
          <span className="pill status-chip num" role="status">
            <span className="spinner" aria-hidden="true" /> Processing on CPU · {(elapsed / 1000).toFixed(1)} s
          </span>
        )}
        {error && !busy && (
          <div className="viewer-error" role="alert">
            <div className="box">
              <span className="err-mark" aria-hidden="true">!</span>
              <strong>{error.message}</strong>
              {error.hint && <span className="secondary">{error.hint}</span>}
              <button className="btn" onClick={onRetry} style={{ marginTop: 8 }}>Try again</button>
            </div>
          </div>
        )}
      </div>

      <div className="viewer-bar">
        <div className="chips scroll" role="group" aria-label="What to compare">
          {quick.map((q) => (
            <button key={q.pair.join()} className="chip" aria-pressed={!custom && isQuick(q.pair)}
              onClick={() => { setCustom(false); onPair(q.pair); }}>
              {SHORT[q.pair[0]]} <span className="vs">vs</span> {SHORT[q.pair[1]]}
            </button>
          ))}
          <button className="chip" aria-pressed={custom || !onQuick} onClick={() => setCustom((c) => !c)} aria-expanded={custom}>
            Custom
          </button>
        </div>
        {custom && (
          <div className="custom-pair">
            <label className="sr-only" htmlFor="left-sel">Left image</label>
            <select id="left-sel" className="select" value={left} onChange={(e) => onPair([e.target.value as ImageKey, right])}>
              {keys.map((k) => <option key={k} value={k}>{LABEL[k]}</option>)}
            </select>
            <button className="btn icon" aria-label="Swap left and right" title="Swap" onClick={() => onPair([right, left])}>
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M2 5h11M10 2l3 3-3 3M14 11H3M6 8l-3 3 3 3" fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>
            </button>
            <label className="sr-only" htmlFor="right-sel">Right image</label>
            <select id="right-sel" className="select" value={right} onChange={(e) => onPair([left, e.target.value as ImageKey])}>
              {keys.map((k) => <option key={k} value={k}>{LABEL[k]}</option>)}
            </select>
          </div>
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ loupe */
const ZOOMS: Zoom[] = [1, 2, 4];

export function Loupe({ result, lens, zoom, onZoom, onTile }: {
  result: ProcessResponse | null; lens: { x: number; y: number }; zoom: Zoom; onZoom: (z: Zoom) => void;
  onTile: (t: { w: number; h: number }) => void;
}) {
  const tileRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = tileRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => onTile({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, [onTile]);

  const W = result?.input.width ?? 3;
  const H = result?.input.height ?? 2;
  const hasGt = !!result?.input.gt_url;
  const keys: ImageKey[] = ["traditional", "hybrid", "learned", ...(hasGt ? (["gt"] as ImageKey[]) : [])];
  const [tw, th] = [tileRef.current?.clientWidth ?? 200, tileRef.current?.clientHeight ?? 133];
  const style = (k: ImageKey) => {
    const url = result ? urlOf(result, k) : null;
    const cx = lens.x * W, cy = lens.y * H;
    return url
      ? { backgroundImage: `url(${url})`, backgroundSize: `${W * zoom}px ${H * zoom}px`, backgroundPosition: `${tw / 2 - cx * zoom}px ${th / 2 - cy * zoom}px` }
      : {};
  };
  return (
    <section className="loupe-wrap" id="loupe" aria-labelledby="loupe-h">
      <div className="loupe-head">
        <h3 id="loupe-h">Pixel loupe <span className="muted">· the same spot in every pipeline</span></h3>
        <div className="chips" role="group" aria-label="Loupe zoom">
          {ZOOMS.map((z) => (
            <button key={z} className="chip mini num" aria-pressed={zoom === z} onClick={() => onZoom(z)}>{z === 1 ? "1:1" : `${z}×`}</button>
          ))}
        </div>
      </div>
      <div className="loupe" style={{ ["--n" as string]: keys.length }}>
        {keys.map((k, i) => {
          const m = k === "gt" || k === "input" ? null : result?.outputs[k]?.metrics;
          return (
            <figure key={k} className={cls(k)}>
              <div ref={i === 0 ? tileRef : undefined} className={"lens" + (result ? "" : " skel")} style={style(k)}
                role="img" aria-label={`${LABEL[k]}, magnified`} />
              <figcaption>
                <span><i className="sw" />{SHORT[k]}</span>
                {m && <span className="num">{fixed(m.psnr, 1)} dB</span>}
              </figcaption>
            </figure>
          );
        })}
      </div>
      <p className="hint">Tap or move over the photo to choose the spot.</p>
    </section>
  );
}
