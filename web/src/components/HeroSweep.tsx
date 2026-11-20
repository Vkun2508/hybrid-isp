import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import type { ProcessResponse } from "../api";
import { fixed } from "../format";
import { drawMosaic, drawShimmer } from "../mosaic";

type Phase = "wait" | "a" | "b" | "done";

// timeline (ms): mosaic hold → sweep 1 (classical) → beat → sweep 2 (hybrid) easing to a stop at the slider
const A0 = 500, A1 = 2000, B0 = 2500, B1 = 3700, REST = 50;
const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const easeOut = (t: number) => 1 - Math.pow(1 - t, 4);
const clamp01 = (t: number) => Math.min(1, Math.max(0, t));
const still = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * The Overview's signature: the photo opens as the sensor's RGGB mosaic, a grey line sweeps it into the
 * traditional pipeline's result, then a blue line sweeps the hybrid's result back in and stops halfway,
 * where it becomes the comparison slider. Any touch during the sweep jumps to the end and follows the finger.
 */
export function HeroSweep({ shot, caption, captionShort, onDone, children }: {
  shot: ProcessResponse | null; caption: string; captionShort?: string; onDone?: () => void; children?: ReactNode;
}) {
  const root = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const sensor = useRef<HTMLImageElement | null>(null);
  const raf = useRef(0);
  const b = useRef(100);
  const drag = useRef(false);
  const [phase, setPhase] = useState<Phase>("wait");
  const [mosaicReady, setMosaicReady] = useState(false);

  const cell = () => (window.matchMedia("(max-width: 999px)").matches ? 5 : 6);
  const setVars = (a: number, bb: number) => {
    b.current = bb;
    root.current?.style.setProperty("--a", `${a}%`);
    root.current?.style.setProperty("--b", `${bb}%`);
    root.current?.querySelector(".hs-handle")?.setAttribute("aria-valuenow", String(Math.round(bb)));
  };

  // paint the placeholder at once, and repaint on resize
  const repaint = useCallback(() => {
    const c = canvas.current;
    if (!c) return;
    if (sensor.current) drawMosaic(c, sensor.current, cell());
    else drawShimmer(c, cell());
  }, []);
  useLayoutEffect(() => {
    setVars(0, 100);
    repaint();
    const ro = new ResizeObserver(() => repaint());
    if (root.current) ro.observe(root.current);
    return () => ro.disconnect();
  }, [repaint]);

  const finish = useCallback((at = REST) => {
    cancelAnimationFrame(raf.current);
    setVars(100, at);
    setPhase((p) => { if (p !== "done") onDone?.(); return "done"; });
  }, [onDone]);

  const play = useCallback(() => {
    cancelAnimationFrame(raf.current);
    if (still()) { finish(); return; }
    setVars(0, 100);
    setPhase("a");
    const t0 = performance.now();
    let cur: Phase = "a";
    const tick = (now: number) => {
      const t = now - t0;
      const a = 100 * easeInOut(clamp01((t - A0) / (A1 - A0)));
      const bb = 100 - (100 - REST) * easeOut(clamp01((t - B0) / (B1 - B0)));
      setVars(a, bb);
      const next: Phase = t >= B1 ? "done" : t >= B0 - 60 ? "b" : "a";
      if (next !== cur) { cur = next; if (next === "done") { finish(bb); return; } setPhase(next); }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
  }, [finish]);

  // the shot arrives (images already preloaded): build the real mosaic, then run the sequence once
  useEffect(() => {
    if (!shot) return;
    if (still()) { finish(); return; }
    const im = new Image();
    im.onload = () => { sensor.current = im; repaint(); setMosaicReady(true); play(); };
    im.onerror = () => finish();
    im.src = shot.input.image_url;
    return () => cancelAnimationFrame(raf.current);
  }, [shot?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const pointTo = (e: React.PointerEvent) => {
    const r = root.current?.getBoundingClientRect();
    return r ? Math.min(100, Math.max(0, ((e.clientX - r.left) / r.width) * 100)) : REST;
  };
  const isControl = (t: EventTarget | null) => !!(t as HTMLElement | null)?.closest("a, button:not(.hs-handle), input, select");

  const hyM = shot?.outputs.hybrid.metrics;
  const trM = shot?.outputs.traditional.metrics;

  return (
    <div
      ref={root}
      className={`hero-sweep phase-${phase}` + (mosaicReady ? " has-mosaic" : "")}
      onPointerDown={(e) => {
        if (!shot || isControl(e.target)) return;
        if (phase !== "done") finish(pointTo(e));
        drag.current = true;
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        setVars(100, pointTo(e));
      }}
      onPointerMove={(e) => { if (drag.current) setVars(100, pointTo(e)); }}
      onPointerUp={() => { drag.current = false; }}
      onPointerCancel={() => { drag.current = false; }}
    >
      <canvas ref={canvas} className="hs-mosaic" aria-hidden="true" />
      {shot && (
        <>
          <img className="hs-img hs-trad" src={shot.outputs.traditional.image_url} alt="The traditional pipeline's result: dark and blotchy" draggable={false} />
          <img className="hs-img hs-hyb" src={shot.outputs.hybrid.image_url} alt="The hybrid pipeline's result: bright and clean" draggable={false} />
        </>
      )}

      <span className="hs-line hs-line-a" aria-hidden="true" />
      <span className="hs-line hs-line-b" aria-hidden="true" />
      <span className="hs-pill hs-pill-a" aria-hidden="true"><i className="sw c-traditional" />Classical camera maths</span>
      <span className="hs-pill hs-pill-b" aria-hidden="true"><i className="sw c-hybrid" />+ our 1M-parameter network</span>

      <button
        className="hs-handle" aria-label="Comparison divider: traditional on the left, hybrid on the right" role="slider"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(b.current)} tabIndex={phase === "done" ? 0 : -1}
        onKeyDown={(e) => {
          const step = e.shiftKey ? 10 : 3;
          if (e.key === "ArrowLeft") setVars(100, Math.max(0, b.current - step));
          else if (e.key === "ArrowRight") setVars(100, Math.min(100, b.current + step));
          else return;
          e.preventDefault();
        }}
      >
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path d="M7 4L2 9l5 5M11 4l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="1.8" /></svg>
      </button>

      <p className="hs-caption"><span className="hs-live" aria-hidden="true" /><span className="cap-long">{caption}</span><span className="cap-short" aria-hidden="true">{captionShort ?? caption}</span></p>
      <span className="hs-corner left"><i className="sw c-traditional" />Traditional{trM && <b className="num">{fixed(trM.psnr, 1)} dB</b>}</span>
      <span className="hs-corner right"><i className="sw c-hybrid" />Hybrid{hyM && <b className="num">{fixed(hyM.psnr, 1)} dB</b>}</span>
      <button className="hs-replay" onClick={() => play()} disabled={!shot} aria-label="Replay the develop animation">
        <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 8a5.5 5.5 0 1 0 1.6-3.9M2.5 2.5v3.2h3.2" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
        Replay
      </button>

      <div className="hs-scrim" aria-hidden="true" />
      {children}
    </div>
  );
}
