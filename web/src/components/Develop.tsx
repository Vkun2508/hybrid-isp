import { useEffect, useRef, useState } from "react";
import type { ProcessResponse } from "../api";
import { ms } from "../format";

interface Frame { url: string; title: string; note: string; px?: boolean }

function framesOf(r: ProcessResponse): Frame[] {
  const st = r.outputs.hybrid.stages;
  const find = (re: RegExp) => st.find((s) => re.test(s.name));
  const out: Frame[] = [];
  const raw = st[0];
  if (raw) out.push({ url: raw.image_url, title: "What the sensor records", note: "The RAW Bayer mosaic, zoomed in: one colour per pixel, mostly green, full of noise.", px: true });
  const net = find(/learned core|network/i);
  if (net && !/skipped/i.test(net.name)) out.push({ url: net.image_url, title: "The network cleans the RAW", note: "A 1M-parameter U-Net removes the noise before any colour processing.", px: true });
  const dm = find(/demosaic/i);
  if (dm) out.push({ url: dm.image_url, title: "Classical: demosaic", note: "Exact maths rebuilds full colour at every pixel." });
  const ct = find(/colour \+ tone|colour/i);
  if (ct) out.push({ url: ct.image_url, title: "Classical: colour and tone", note: "Camera colour → sRGB, then the tone curve." });
  out.push({ url: r.outputs.hybrid.image_url, title: "The finished photo", note: `The hybrid's output, in ${ms(r.outputs.hybrid.latency_ms)} on a CPU.` });
  return out;
}

const HOLD = 1700;

/** The signature: a real photo develops from its sensor mosaic, stage by stage, then becomes a before/after slider. */
export function Develop({ result }: { result: ProcessResponse | null }) {
  const still = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [idx, setIdx] = useState(0);
  const [compare, setCompare] = useState(still);
  const [playing, setPlaying] = useState(false);
  const [started, setStarted] = useState(still);
  const box = useRef<HTMLDivElement>(null);
  const frames = result ? framesOf(result) : [];

  // preload every frame so the sequence never waits on the network
  useEffect(() => { frames.forEach((f) => { const im = new Image(); im.src = f.url; }); }, [result?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // start once, when the hero is on screen and the images are known
  useEffect(() => {
    if (started || !result || !box.current) return;
    const io = new IntersectionObserver((es) => {
      if (es.some((e) => e.isIntersecting)) { setStarted(true); setPlaying(true); io.disconnect(); }
    }, { threshold: 0.35 });
    io.observe(box.current);
    return () => io.disconnect();
  }, [started, result]);

  useEffect(() => {
    if (!playing || compare) return;
    const t = setTimeout(() => {
      if (idx < frames.length - 1) setIdx(idx + 1);
      else { setPlaying(false); setCompare(true); }
    }, idx === frames.length - 1 ? HOLD + 300 : HOLD);
    return () => clearTimeout(t);
  }, [playing, idx, compare, frames.length]);

  const replay = () => { setCompare(false); setIdx(0); setPlaying(true); };
  const jump = (i: number) => { setPlaying(false); setCompare(false); setIdx(i); };
  const f = frames[idx];

  return (
    <figure className="develop" ref={box}>
      <div className="dev-frame">
        {!result && <div className="skel" style={{ position: "absolute", inset: 0 }} />}
        {frames.map((fr, i) => (
          <img key={fr.url} src={fr.url} alt={i === idx && !compare ? `${fr.title}: ${fr.note}` : ""} aria-hidden={i !== idx || compare}
            className={"dev-img" + (fr.px ? " px" : "") + (i === idx && !compare ? " on" : "")} draggable={false} />
        ))}
        {result && compare && <MiniCompare left={result.outputs.traditional.image_url} right={result.outputs.hybrid.image_url} />}
        {result && !compare && f && (
          <figcaption className="dev-cap" aria-live="polite">
            <span className="num dev-step">{idx + 1}/{frames.length}</span>
            <b>{f.title}</b>
            <span>{f.note}</span>
          </figcaption>
        )}
      </div>
      <div className="dev-bar">
        <ol className="dev-ticks" aria-label="Development stages">
          {frames.map((fr, i) => (
            <li key={i}>
              <button aria-current={!compare && i === idx ? "step" : undefined} onClick={() => jump(i)} aria-label={`Show stage ${i + 1}: ${fr.title}`}>
                <span className={"tick-bar" + (!compare && i === idx && playing ? " run" : "") + ((compare || i < idx) ? " done" : "")} style={{ ["--hold" as string]: `${HOLD}ms` }} />
              </button>
            </li>
          ))}
          <li>
            <button aria-current={compare ? "step" : undefined} onClick={() => { setPlaying(false); setCompare(true); }} className="tick-cmp">Compare</button>
          </li>
        </ol>
        <button className="btn text small" onClick={replay} disabled={!result}>Replay</button>
      </div>
    </figure>
  );
}

function MiniCompare({ left, right }: { left: string; right: string }) {
  const [split, setSplit] = useState(50);
  const area = useRef<HTMLDivElement>(null);
  const drag = useRef(false);
  const to = (e: React.PointerEvent) => {
    const r = area.current?.getBoundingClientRect();
    if (r) setSplit(Math.min(100, Math.max(0, ((e.clientX - r.left) / r.width) * 100)));
  };
  return (
    <div
      className="mini-compare" ref={area}
      onPointerDown={(e) => { drag.current = true; (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId); to(e); }}
      onPointerMove={(e) => { if (drag.current) to(e); }}
      onPointerUp={() => { drag.current = false; }} onPointerCancel={() => { drag.current = false; }}
    >
      <img src={left} alt="Traditional pipeline result" draggable={false} />
      <img src={right} alt="Hybrid pipeline result" draggable={false} className="layer-b" style={{ ["--split" as string]: `${split}%` }} />
      <span className="split-line" style={{ ["--split" as string]: `${split}%` }} />
      <button
        className="split-handle" style={{ ["--split" as string]: `${split}%` }} role="slider" aria-label="Comparison divider"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(split)}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") setSplit((s) => Math.max(0, s - 4));
          else if (e.key === "ArrowRight") setSplit((s) => Math.min(100, s + 4));
          else return;
          e.preventDefault();
        }}
      >
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path d="M7 4L2 9l5 5M11 4l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="1.8" /></svg>
      </button>
      <span className="corner left c-traditional"><i className="sw" />Traditional</span>
      <span className="corner right c-hybrid"><i className="sw" />Hybrid</span>
    </div>
  );
}
