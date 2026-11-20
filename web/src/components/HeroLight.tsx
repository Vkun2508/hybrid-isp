import { memo, useCallback, useEffect, useLayoutEffect, useRef, useState, type MutableRefObject, type ReactNode } from "react";
import { fixed } from "../format";
import {
  HERO_PIPES, heroImage, heroSensor, heroVariant, lessLight, stopText,
  type HeroData, type HeroPipe, type HeroVariant,
} from "../hero";
import { preload } from "../lab/useLab";
import { drawMosaic, drawShimmer } from "../mosaic";

type Phase = "load" | "sweep" | "hold" | "dim" | "done";

// timeline (ms): sensor mosaic → develop sweep into the three strips → hold at full light → the light drops to −3 EV
const S0 = 300, S1 = 1300, D0 = 2000, D1 = 4400, TARGET = 3, SNAP_MS = 160;
const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const easeOut = (t: number) => 1 - Math.pow(1 - t, 4);
const clamp01 = (t: number) => Math.min(1, Math.max(0, t));
const still = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const cell = () => (window.matchMedia("(max-width: 999px)").matches ? 5 : 6);

const NAME: Record<HeroPipe, [string, string]> = {
  traditional: ["Camera maths only", "Maths"],
  learned: ["Network only", "Network"],
  hybrid: ["Hybrid", "Hybrid"],
};

const Moon = () => (
  <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path d="M14.5 11.2A6 6 0 0 1 6.8 3.5a6 6 0 1 0 7.7 7.7z" fill="currentColor" /></svg>
);
const Sun = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
    <circle cx="10" cy="10" r="3.6" fill="currentColor" />
    <path d="M10 1.8v2.4M10 15.8v2.4M1.8 10h2.4M15.8 10h2.4M4.2 4.2l1.7 1.7M14.1 14.1l1.7 1.7M4.2 15.8l1.7-1.7M14.1 5.9l1.7-1.7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);

/** The 15 stacked images. Memoised: the dial re-renders the labels, never these (opacity is set through refs). */
const Strips = memo(function Strips({ hero, variant, imgs }: {
  hero: HeroData; variant: HeroVariant; imgs: MutableRefObject<(HTMLImageElement | null)[]>;
}) {
  const n = hero.levels.length;
  return (
    <>
      {HERO_PIPES.map((p, i) => (
        <div className={`hl-strip s${i}`} key={p} aria-hidden="true">
          {hero.levels.map((_, k) => (
            <img
              key={k}
              ref={(el) => { imgs.current[i * n + k] = el; }}
              className={`hl-img k${k}`}
              src={heroImage(variant, p, k)}
              alt=""
              draggable={false}
              decoding="async"
            />
          ))}
        </div>
      ))}
    </>
  );
});

/**
 * The Overview's signature. One real RAW, cut into three aligned strips: camera maths only, the network only, and
 * the hybrid. It opens as the sensor's RGGB mosaic, develops into the strips at full light, then the light dial
 * drops to −3 EV: two strips sink into black while the hybrid holds, and the scores click down stop by stop.
 * After that the visitor drives the dial. Numbers at rest are always measured values (the dial snaps to stops).
 */
export function HeroLight({ hero, copy, onDone, onStop, onRunLive }: {
  hero: HeroData | null;
  copy: ReactNode;
  onDone?: () => void;
  onStop?: (stop: number) => void;
  onRunLive?: (stop: number) => void;
}) {
  const root = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const range = useRef<HTMLInputElement>(null);
  const imgs = useRef<(HTMLImageElement | null)[]>([]); // index: pipe * levels + level
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const sensor = useRef<HTMLImageElement | null>(null);
  const raf = useRef(0);
  const dRef = useRef(0);
  const stopRef = useRef(0);
  const user = useRef(false);
  const doneOnce = useRef(false);
  const started = useRef(false);
  const [variant, setVariant] = useState<HeroVariant>(heroVariant);
  const [phase, setPhase] = useState<Phase>("load");
  const [stop, setStop] = useState(0);
  const [touched, setTouched] = useState(false);
  const [said, setSaid] = useState("");
  const n = hero?.levels.length ?? 5;
  const maxD = n - 1;

  // ---- per-frame writers (no React renders except when the stop changes)
  const setA = (a: number) => {
    root.current?.style.setProperty("--a", `${a}%`);
    labels.current.forEach((el, i) => { if (el) el.style.opacity = String(clamp01((a - i * 33.333 - 6) / 14)); });
  };
  const setD = (d: number) => {
    dRef.current = d;
    for (let p = 0; p < 3; p++) {
      for (let k = 1; k < n; k++) {
        const el = imgs.current[p * n + k];
        if (el) el.style.opacity = String(clamp01(d - k + 1));
      }
    }
    if (range.current) range.current.value = String(-d);
    root.current?.style.setProperty("--seam", String(Math.min(1, d / 2)));
    const s = Math.round(d);
    if (s !== stopRef.current) {
      stopRef.current = s;
      setStop(s);
      onStop?.(s);
      if (user.current && "vibrate" in navigator) navigator.vibrate(8);
    }
  };
  const announce = useCallback(() => {
    if (!hero) return;
    const s = stopRef.current;
    const v = (p: HeroPipe) => fixed(hero.metrics[p][s]?.psnr, 1);
    setSaid(`${stopText(s)}, ${lessLight(s)}: camera maths only ${v("traditional")} dB, network only ${v("learned")} dB, hybrid ${v("hybrid")} dB.`);
  }, [hero]);

  // ---- mosaic: placeholder at once, the real sensor view once loaded; repaint on resize
  const repaint = useCallback(() => {
    const c = canvas.current;
    if (!c) return;
    if (sensor.current) drawMosaic(c, sensor.current, cell());
    else drawShimmer(c, cell());
  }, []);
  useLayoutEffect(() => {
    setA(0);
    setD(0);
    repaint();
    const ro = new ResizeObserver(() => repaint());
    if (root.current) ro.observe(root.current);
    const mq = window.matchMedia("(max-width: 699px)");
    const onMq = () => setVariant(heroVariant());
    mq.addEventListener("change", onMq);
    return () => { ro.disconnect(); mq.removeEventListener("change", onMq); cancelAnimationFrame(raf.current); };
  }, [repaint]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- timeline
  const markDone = useCallback(() => {
    setPhase("done");
    if (!doneOnce.current) { doneOnce.current = true; onDone?.(); }
  }, [onDone]);
  const finish = useCallback((at = TARGET) => {
    cancelAnimationFrame(raf.current);
    setA(100);
    setD(at);
    markDone();
    announce();
  }, [markDone, announce]); // eslint-disable-line react-hooks/exhaustive-deps

  const play = useCallback(() => {
    cancelAnimationFrame(raf.current);
    user.current = false;
    if (still()) { finish(); return; }
    setA(0);
    setD(0);
    setPhase("sweep");
    const t0 = performance.now();
    let cur: Phase = "sweep";
    const tick = (now: number) => {
      const t = now - t0;
      setA(100 * easeInOut(clamp01((t - S0) / (S1 - S0))));
      setD(TARGET * easeInOut(clamp01((t - D0) / (D1 - D0))));
      if (t >= D1) { finish(TARGET); return; }
      const next: Phase = t < S1 ? "sweep" : t < D0 ? "hold" : "dim";
      if (next !== cur) { cur = next; setPhase(next); }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
  }, [finish]); // eslint-disable-line react-hooks/exhaustive-deps

  // the visitor grabbed the dial mid-sequence: stop the timeline where it is
  const takeOver = () => {
    cancelAnimationFrame(raf.current);
    user.current = true;
    setTouched(true);
    if (phase !== "done") { setA(100); markDone(); }
  };
  const snapTo = (target: number) => {
    cancelAnimationFrame(raf.current);
    const to = Math.min(maxD, Math.max(0, target));
    if (still()) { setD(to); announce(); return; }
    const from = dRef.current, t0 = performance.now();
    const tick = (now: number) => {
      const t = clamp01((now - t0) / SNAP_MS);
      setD(from + (to - from) * easeOut(t));
      if (t < 1) raf.current = requestAnimationFrame(tick);
      else announce();
    };
    raf.current = requestAnimationFrame(tick);
  };
  const release = () => { if (user.current) snapTo(Math.round(dRef.current)); };

  // native "change" fires on release everywhere (iOS included), even when pointer events don't
  useEffect(() => {
    const el = range.current;
    if (!el) return;
    const onChange = () => release();
    el.addEventListener("change", onChange);
    return () => el.removeEventListener("change", onChange);
  }); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- load: the sensor view and levels 0–3 of this variant, then run the sequence once
  useEffect(() => {
    if (!hero) return;
    let off = false;
    sensor.current = null;
    const im = new Image();
    const sensorReady = new Promise<boolean>((res) => { im.onload = () => res(true); im.onerror = () => res(false); im.src = heroSensor(variant); });
    const urls = HERO_PIPES.flatMap((p) => [0, 1, 2, 3].filter((k) => k < n).map((k) => heroImage(variant, p, k)));
    // decode the strips' own <img>s up front, so a level never stalls a frame the first time it fades in
    const decoded = () => Promise.race([
      Promise.all(imgs.current.filter((el): el is HTMLImageElement => !!el && !el.src.endsWith("_4.webp")).map((el) => el.decode().catch(() => undefined))),
      new Promise((res) => setTimeout(res, 1500)),
    ]);
    Promise.all([sensorReady, preload(urls, 6000).then(decoded)]).then(([ok]) => {
      if (off) return;
      if (ok) { sensor.current = im; repaint(); }
      if (!started.current) { started.current = true; play(); }
    });
    return () => { off = true; };
  }, [hero, variant]); // eslint-disable-line react-hooks/exhaustive-deps

  const isControl = (t: EventTarget | null) => !!(t as HTMLElement | null)?.closest("a, button, input, select, .term, .hl-dial");
  const playing = phase === "sweep" || phase === "hold" || phase === "dim";

  return (
    <div
      ref={root}
      className={`hero-light phase-${phase}`}
      onPointerDown={(e) => { if (playing && !isControl(e.target)) finish(TARGET); }}
    >
      {hero && <Strips hero={hero} variant={variant} imgs={imgs} />}
      <span className="hl-seam s1" aria-hidden="true" />
      <span className="hl-seam s2" aria-hidden="true" />
      <canvas ref={canvas} className="hl-mosaic" aria-hidden="true" />
      <span className="hl-sweep" aria-hidden="true" />
      <div className="hl-scrim" aria-hidden="true" />

      <div className="hl-labels" aria-hidden="true">
        {HERO_PIPES.map((p, i) => {
          const m = hero?.metrics[p];
          const v = m?.[stop]?.psnr;
          const base = m?.[0]?.psnr;
          const delta = v !== undefined && base !== undefined ? v - base : null;
          return (
            <div key={p} ref={(el) => { labels.current[i] = el; }} className={`hl-label c-${p}`}>
              <span className="hl-name">
                <i className={`sw c-${p}`} />
                <span className="cap-long">{NAME[p][0]}</span><span className="cap-short">{NAME[p][1]}</span>
                {p === "hybrid" && <em>ours</em>}
              </span>
              <b className="hl-db" key={stop}>{fixed(v, 1)}<small>dB</small></b>
              <span className="hl-delta">
                {stop === 0 || delta === null ? "at full light" : `${delta < 0 ? "▼" : "▲"} ${Math.abs(delta).toFixed(1)} dB`}
              </span>
            </div>
          );
        })}
      </div>

      <div className="hl-foot">
        <div className="hl-copy">{copy}</div>
        <div className="hl-dial" role="group" aria-label="Light">
          <div className="hl-read">
            <span className="hl-ev">{stopText(stop)}</span>
            <span className="hl-less">{lessLight(stop)}</span>
            <button className="hl-replay" onClick={() => play()} disabled={!hero || playing} aria-label="Replay the animation">
              <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 8a5.5 5.5 0 1 0 1.6-3.9M2.5 2.5v3.2h3.2" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
              <span className="cap-long">Replay</span>
            </button>
          </div>
          <div className="hl-track">
            <Moon />
            <div className="hl-range-wrap">
              <input
                ref={range}
                className={"hl-range" + (phase === "done" && !touched ? " pulse" : "")}
                type="range" min={-maxD} max={0} step={0.01} defaultValue={0}
                aria-label="Light, in stops below the original exposure"
                aria-valuetext={`${stopText(stop)}, ${lessLight(stop)}`}
                disabled={!hero}
                onPointerDown={takeOver}
                onInput={(e) => { takeOver(); setD(-Number(e.currentTarget.value)); }}
                onPointerUp={release}
                onPointerCancel={release}
                onKeyDown={(e) => {
                  const s = Math.round(dRef.current);
                  const to = e.key === "ArrowLeft" || e.key === "ArrowDown" || e.key === "PageDown" ? s + 1
                    : e.key === "ArrowRight" || e.key === "ArrowUp" || e.key === "PageUp" ? s - 1
                    : e.key === "Home" ? maxD : e.key === "End" ? 0 : null;
                  if (to === null) return;
                  e.preventDefault();
                  takeOver();
                  snapTo(to);
                }}
              />
              <div className="hl-ticks" aria-hidden="true">
                {Array.from({ length: n }, (_, i) => (
                  <span key={i} style={{ left: `${(i / maxD) * 100}%` }} className={maxD - i === stop ? "on" : ""}>
                    <i />{i === maxD ? "0" : `−${maxD - i}`}
                  </span>
                ))}
              </div>
            </div>
            <Sun />
          </div>
          <div className="hl-dial-foot">
            <span className="hl-hint">{touched ? `${hero?.camera ?? "Real"} RAW · not in the training set` : "Drag to change the light"}</span>
            <button className="btn text hl-live" onClick={() => onRunLive?.(stop)} disabled={!hero}>Run it live →</button>
          </div>
        </div>
      </div>
      <p className="sr-only" aria-live="polite">{said}</p>
    </div>
  );
}
