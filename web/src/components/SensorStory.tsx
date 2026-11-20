import { useEffect, useRef, useState } from "react";
import { fixed } from "../format";
import type { StoryScene } from "../story/scene";

type Meta = {
  camera: string; iso: number; ev: number; photosites: number; gain: number;
  psnr: { traditional: number; learned: number; hybrid: number };
  classical_gain: { psnr: number };
  stats: { noisy_jitter: number; denoised_jitter: number; gained_noisy_jitter: number; gained_jitter: number };
};

const clamp01 = (t: number) => Math.min(1, Math.max(0, t));
const smooth = (t: number) => t * t * (3 - 2 * t);
const ramp = (p: number, a: number, b: number) => smooth(clamp01((p - a) / (b - a)));
/** opacity of a block shown between a and b (fading over f at each end) */
const span = (p: number, a: number, b: number, f = 0.04) => Math.min(ramp(p, a, a + f), 1 - ramp(p, b - f, b));

const STEPS = [
  { who: "network", at: [0.12, 0.39] },
  { who: "maths", at: [0.39, 0.63] },
  { who: "maths", at: [0.63, 0.86] },
] as const;

/**
 * The Overview's signature: a pinned, scroll-driven WebGL stage of one real RAW's photosites. Scroll (or "Watch it
 * develop") runs the hybrid pipeline in its real order: the network stills the noise, exact maths raises the clean
 * surface, and the field tilts flat into the photo. It lands on the proof: the same RAW through a classical pipeline,
 * split against the hybrid through the chess piece, with a divider the visitor can drag. Any failure (no WebGL,
 * lost context, missing data) calls onFallback, and the Overview shows the "Lights down" hero instead.
 */
export function SensorStory({ onFallback, onTour, onTab, onRunLive, onDone }: {
  onFallback: () => void; onTour: () => void; onTab: (t: "lab") => void; onRunLive: () => void; onDone?: () => void;
}) {
  const sec = useRef<HTMLElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const photos = useRef<HTMLDivElement>(null);
  const intro = useRef<HTMLDivElement>(null);
  const end = useRef<HTMLDivElement>(null);
  const caps = useRef<(HTMLLIElement | null)[]>([]);
  const rail = useRef<HTMLDivElement>(null);
  const handle = useRef<HTMLButtonElement>(null);
  const sc = useRef<StoryScene | null>(null);
  const auto = useRef(0);
  const doneOnce = useRef(false);
  const rest = useRef(0.3); // where the split settles: through the chess piece
  const manual = useRef<number | null>(null); // the visitor moved the divider
  const lastP = useRef(0);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);

  const progress = () => {
    const s = sec.current, st = stage.current;
    if (!s || !st) return 0;
    const r = s.getBoundingClientRect();
    return clamp01(-r.top / Math.max(1, r.height - st.clientHeight));
  };
  const show = (el: HTMLElement | null, o: number, dy = 0) => {
    if (!el) return;
    el.style.opacity = String(o);
    el.style.visibility = o < 0.02 ? "hidden" : "visible";
    if (dy) el.style.transform = `translate3d(0, ${(1 - o) * dy}px, 0)`;
  };
  const setSplit = (p: number) => {
    // the line already sits through the chess piece (it has since step 1); the visitor can move it at the end
    const v = p >= 0.96 && manual.current !== null ? manual.current : rest.current;
    photos.current?.style.setProperty("--split", `${(v * 100).toFixed(2)}%`);
    handle.current?.setAttribute("aria-valuenow", String(Math.round(v * 100)));
  };
  // scroll → progress → scene + overlays (refs only; no React render per frame)
  const apply = (p: number) => {
    lastP.current = p;
    sc.current?.setProgress(p);
    show(intro.current, 1 - ramp(p, 0.03, 0.1), -16);
    STEPS.forEach((s, i) => show(caps.current[i], span(p, s.at[0], s.at[1]), 18));
    show(end.current, ramp(p, 0.9, 0.96), 18);
    if (photos.current) {
      photos.current.style.opacity = String(ramp(p, 0.83, 0.9));
      photos.current.style.setProperty("--tags", String(ramp(p, 0.86, 0.92)));
      photos.current.classList.toggle("live", p >= 0.96);
    }
    setSplit(p);
    if (rail.current) {
      show(rail.current, 1 - ramp(p, 0.84, 0.88));
      rail.current.style.setProperty("--p", String(p));
      rail.current.dataset.step = String(p < 0.12 ? 0 : p < 0.39 ? 1 : p < 0.63 ? 2 : p < 0.86 ? 3 : 4);
    }
    if (p > 0.9 && !doneOnce.current) { doneOnce.current = true; onDone?.(); }
  };
  const frame = () => {
    const t = sc.current?.top();
    if (!t || !photos.current) return;
    photos.current.style.setProperty("--pos", `${(t.posX * 100).toFixed(2)}%`);
    rest.current = Math.min(0.7, Math.max(0.3, t.piece));
    stage.current?.classList.toggle("split-left", rest.current < 0.42);
  };

  // load data + scene (lazy chunk); any failure → fallback
  useEffect(() => {
    let off = false;
    const c = canvas.current;
    if (!c) return;
    // three.js needs WebGL2; probe on a scratch canvas so the real one stays free for the renderer
    if (!document.createElement("canvas").getContext("webgl2")) { onFallback(); return; }
    Promise.all([
      import("../story/scene"),
      fetch("/story/meta.json").then((r) => (r.ok ? r.json() : Promise.reject(new Error("meta")))),
    ]).then(async ([mod, m]) => {
      const field = await mod.loadField("/story/field.bin");
      if (off) return;
      const s = mod.createScene(c, field, { onLost: onFallback });
      sc.current = s;
      setMeta(m as Meta);
      frame();
      apply(progress());
      s.start();
      setReady(true);
    }).catch(() => { if (!off) onFallback(); });
    return () => { off = true; sc.current?.dispose(); sc.current = null; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // scroll, resize, visibility
  useEffect(() => {
    let pending = 0;
    const onScroll = () => { if (!pending) pending = requestAnimationFrame(() => { pending = 0; apply(progress()); }); };
    const onResize = () => { sc.current?.resize(); frame(); onScroll(); };
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onResize);
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting && !document.hidden) sc.current?.start(); else sc.current?.stop(); });
    if (sec.current) io.observe(sec.current);
    const onVis = () => { if (document.hidden) sc.current?.stop(); else if (sec.current && sec.current.getBoundingClientRect().bottom > 0) sc.current?.start(); };
    document.addEventListener("visibilitychange", onVis);
    apply(progress());
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVis);
      io.disconnect();
      cancelAnimationFrame(pending);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // "Watch it develop": glide through the section; any wheel, touch or key hands control back
  const stopAuto = () => { cancelAnimationFrame(auto.current); auto.current = 0; setPlaying(false); };
  const play = () => {
    const s = sec.current, st = stage.current;
    if (!s || !st) return;
    stopAuto();
    const top = s.getBoundingClientRect().top + window.scrollY;
    const from = window.scrollY, to = top + s.offsetHeight - st.clientHeight;
    const D = 10500, t0 = performance.now();
    setPlaying(true);
    const tick = (now: number) => {
      const t = clamp01((now - t0) / D);
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      window.scrollTo({ top: from + (to - from) * e, behavior: "instant" as ScrollBehavior });
      if (t < 1) auto.current = requestAnimationFrame(tick); else stopAuto();
    };
    auto.current = requestAnimationFrame(tick);
  };
  useEffect(() => {
    if (!playing) return;
    const cancel = () => stopAuto();
    window.addEventListener("wheel", cancel, { passive: true });
    window.addEventListener("touchstart", cancel, { passive: true });
    window.addEventListener("keydown", cancel);
    return () => { window.removeEventListener("wheel", cancel); window.removeEventListener("touchstart", cancel); window.removeEventListener("keydown", cancel); };
  }, [playing]);

  // the divider: drag it, or use the arrow keys
  const dragTo = (clientX: number) => {
    const r = stage.current?.getBoundingClientRect();
    if (!r) return;
    manual.current = Math.min(0.98, Math.max(0.02, (clientX - r.left) / r.width));
    setSplit(lastP.current);
  };

  const ps = meta?.psnr;
  const st = meta?.stats;
  return (
    <section className={"story" + (ready ? " ready" : "")} ref={sec} aria-labelledby="story-h">
      <div className="story-stage" ref={stage}>
        <canvas ref={canvas} className="story-canvas" aria-hidden="true" />

        <div className="story-photos" ref={photos}>
          <img className="story-photo trad" src={ready ? "/hero/d/classical_gain_3.webp" : undefined} alt="The same RAW through a classical pipeline brightened the same way: grainy, with colour blotches" decoding="async" />
          <img className="story-photo hyb" src={ready ? "/hero/d/hybrid_3.webp" : undefined} alt="The hybrid pipeline's result: the chess pieces clearly visible" decoding="async" />
          <span className="story-split" aria-hidden="true" />
          <span className="story-tag left" aria-hidden="true">Classical, same ×8 gain<b>{fixed(meta?.classical_gain.psnr, 1)} dB</b></span>
          <span className="story-tag right" aria-hidden="true">Hybrid<b>{fixed(ps?.hybrid, 1)} dB</b></span>
          <button
            ref={handle} className="story-handle" role="slider" aria-label="Divider: classical pipeline on the left, hybrid on the right"
            aria-valuemin={0} aria-valuemax={100} aria-valuenow={30}
            onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); dragTo(e.clientX); }}
            onPointerMove={(e) => { if (e.currentTarget.hasPointerCapture(e.pointerId)) dragTo(e.clientX); }}
            onKeyDown={(e) => {
              const cur = manual.current ?? rest.current, d = e.shiftKey ? 0.1 : 0.03;
              if (e.key === "ArrowLeft") manual.current = Math.max(0.02, cur - d);
              else if (e.key === "ArrowRight") manual.current = Math.min(0.98, cur + d);
              else return;
              e.preventDefault();
              setSplit(lastP.current);
            }}
          >
            <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path d="M7 4L2 9l5 5M11 4l5 5-5 5" fill="none" stroke="currentColor" strokeWidth="1.8" /></svg>
          </button>
        </div>
        <div className="story-shade" aria-hidden="true" />

        <div className="story-intro" ref={intro}>
          <h1 id="story-h">Night photos start as&nbsp;noise.</h1>
          <p className="story-lead">
            <span className="cap-long">These are {meta ? meta.photosites.toLocaleString("en") : "24,576"} real photosites from a phone RAW (Nexus&nbsp;6 at ISO&nbsp;3200, SIDD dataset), scaled three stops darker. Each one sees a single colour. Scroll, and our hybrid pipeline develops them.</span>
            <span className="cap-short">Real photosites from a low-light phone RAW (SIDD, Nexus&nbsp;6), scaled three stops darker. Scroll to develop them.</span>
          </p>
          <div className="story-actions">
            <button className="btn primary large" onClick={play}>
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 3v10l8-5z" fill="currentColor" /></svg>
              Watch it develop
            </button>
            <button className="btn text story-skip" onClick={onTour}>Start the guided demo</button>
          </div>
          <p className="story-kicker">Hybrid ISP · Computer Vision Hackathon 2026 · Problem P2</p>
        </div>

        <ol className="story-caps" aria-label="What the pipeline does">
          <li ref={(el) => { caps.current[0] = el; }} className="story-cap network">
            <b>The <em>network</em> cleans the RAW.</b>
            <span className="story-body">Right of the line, a 1M-parameter network has taken the noise out of the sensor data itself. Left of it, the RAW as captured.</span>
            {st && <span className="story-read">Difference between neighbouring photosites: <strong>{st.noisy_jitter.toFixed(1)}</strong> → <strong className="net">{st.denoised_jitter.toFixed(1)}</strong> levels</span>}
          </li>
          <li ref={(el) => { caps.current[1] = el; }} className="story-cap maths">
            <b>Exact <em>maths</em> brightens both, ×8.</b>
            <span className="story-body">The same gain and white balance on both sides. Brightened noise turns into spikes on the left; the cleaned RAW rises as a landscape on the right.</span>
            {st && <span className="story-read">After brightening: <strong>{st.gained_noisy_jitter.toFixed(1)}</strong> left, <strong className="net">{st.gained_jitter.toFixed(1)}</strong> right</span>}
          </li>
          <li ref={(el) => { caps.current[2] = el; }} className="story-cap maths">
            <b><em>Maths</em> develops the photo.</b>
            <span className="story-body">Demosaic, colour and tone, identical on both sides of the line.</span>
          </li>
        </ol>

        <div className="story-rail" ref={rail} aria-hidden="true">
          <span>RAW</span><span className="network">Network</span><span>Maths</span><span>Photo</span>
        </div>

        <div className="story-end" ref={end}>
          <h2>Maths where it’s exact. <span>A network where it isn’t.</span></h2>
          <p className="story-proof">
            <b>{fixed(ps?.hybrid, 1)}<small>dB</small></b>
            <span>on this image at −3 EV, against {fixed(meta?.classical_gain.psnr, 1)}&nbsp;dB for a textbook classical pipeline brightened the same way. PSNR of the rendered photo against the clean RAW rendered identically. Drag the line.</span>
          </p>
          <div className="story-ctas">
            <button className="btn primary large" onClick={onTour}>
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 3v10l8-5z" fill="currentColor" /></svg>
              Start the guided demo
            </button>
            <span className="story-links">
              <button className="btn text" onClick={onRunLive}>Run it live</button>
              <button className="btn text" onClick={() => onTab("lab")}>Try your own photo</button>
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
