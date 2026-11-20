import { type ReactNode } from "react";
import type { ApiError, Benchmarks } from "../api";
import { cls, fixed } from "../format";
import { DeltaBars, HBars, LineChart, SERIES_NAME } from "./charts";
import { Term } from "./Term";

type Bench = { b: Benchmarks | null; err: ApiError | null; reload: () => void };

const SECTIONS: [string, string][] = [
  ["quality", "Quality"], ["robustness", "Robustness"], ["efficiency", "Efficiency"], ["router", "Router"],
  ["ablation", "Ablation"], ["phones", "Per phone"], ["phone-raw", "Phone RAW"], ["failures", "Failures"],
];

function Sec({ id, n, kicker, title, sub, children }: { id: string; n: number; kicker: string; title: ReactNode; sub?: ReactNode; children: ReactNode }) {
  return (
    <section className="ev-sec" id={id} aria-labelledby={`${id}-h`}>
      <div className="ev-head">
        <p className="ev-kicker"><span className="num">{String(n).padStart(2, "0")}</span> {kicker}</p>
        <h2 id={`${id}-h`}>{title}</h2>
        {sub && <p className="ev-sub">{sub}</p>}
      </div>
      {children}
    </section>
  );
}

const Table = ({ children, label = "Show table" }: { children: ReactNode; label?: string }) => (
  <details className="more"><summary>{label}</summary>{children}</details>
);

export function Results({ bench }: { bench: Bench }) {
  const { b, err, reload } = bench;

  if (err)
    return (
      <main className="results" id="main">
        <section className="card" role="alert">
          <h2>Couldn't load the results</h2>
          <p className="secondary" style={{ margin: "8px 0 12px" }}>{err.message} {err.hint}</p>
          <button className="btn" onClick={reload}>Try again</button>
        </section>
      </main>
    );
  if (b === null)
    return (
      <main className="results" id="main" aria-busy="true">
        <div className="ev-intro"><h1>The evidence</h1><div className="skel" style={{ height: 14, width: 360, maxWidth: "100%", marginTop: 12 }} /></div>
        {[220, 260, 200].map((h, i) => <section className="card" key={i} aria-hidden="true"><div className="skel" style={{ height: h }} /></section>)}
      </main>
    );

  const q = (p: string) => b.quality?.find((x) => x.pipeline === p);
  const hy = q("hybrid"), tr = q("traditional");
  const scored = (b.quality ?? []).filter((x) => x.pipeline !== "input");

  // quality: which of the four scores does the hybrid win?
  const wins: string[] = [];
  const losses: string[] = [];
  if (hy && scored.length) {
    const check = (k: "psnr" | "ssim" | "delta_e" | "lpips", name: string, lower: boolean) => {
      const vals = scored.map((x) => x[k]).filter((v): v is number => typeof v === "number");
      const v = hy[k];
      if (typeof v !== "number" || !vals.length) return;
      const best = lower ? Math.min(...vals) : Math.max(...vals);
      if (v === best) wins.push(name);
      else {
        const who = scored.find((x) => x[k] === best);
        losses.push(`${SERIES_NAME[who?.pipeline ?? ""] ?? who?.pipeline} edges it on ${name} (${fixed(best, 3)} vs ${fixed(v, 3)})`);
      }
    };
    check("psnr", "PSNR", false); check("ssim", "SSIM", false); check("delta_e", "colour error", true); check("lpips", "LPIPS", true);
  }
  const nScores = wins.length + losses.length;

  // efficiency
  const fp32 = b.efficiency?.find((e) => /fp32/i.test(e.variant));
  const int8 = b.efficiency?.find((e) => /int8/i.test(e.variant));
  const int8Cost = (() => {
    const full = b.ablation?.find((a) => /full/i.test(a.variant));
    const i8 = b.ablation?.find((a) => /int8/i.test(a.variant));
    return full && i8 ? full.psnr - i8.psnr : null;
  })();

  // router
  const modes = b.router?.modes ?? [];
  const adaptive = modes.find((m) => /adaptive/i.test(m.mode));
  const always = modes.find((m) => /always/i.test(m.mode));

  // ablation
  const full = b.ablation?.find((a) => /full/i.test(a.variant));
  const deltas = full ? (b.ablation ?? []).filter((a) => a !== full).map((a) => ({ key: a.variant, label: a.variant, delta: a.psnr - full.psnr })).sort((x, y) => x.delta - y.delta) : [];
  const top2 = deltas.slice(0, 2).map((d) => d.label.replace(/^No /, "").replace(/ \(.*\)$/, ""));
  const tiny = deltas.filter((d) => Math.abs(d.delta) < 0.1).length;

  // per phone
  const phones = b.breakdown?.by_phone ?? [];
  const phoneWins = phones.filter((g) => g.hybrid.psnr >= Math.max(g.traditional.psnr, g.learned.psnr)).length;
  const raws = b.phone_raw ?? [];
  const passed = raws.filter((p) => p.passed).length;

  let n = 0;
  return (
    <main className="results" id="main">
      <header className="ev-intro">
        <h1>The evidence</h1>
        <p className="lead">
          Every number here comes from scenes held out of training{b.dataset ? <>: {b.dataset.name}, test scenes {b.dataset.test_scenes.join(" and ")}, {b.dataset.n_images} images from 5 phones</> : null}.
          {b.updated && <span className="muted"> Updated {b.updated}.</span>}
        </p>
        {b.status !== "final" && (
          <p className="pill warning" style={{ justifySelf: "start" }}><span className="dot" />{b.status === "pending" ? "Evaluation pending" : "Provisional numbers"}</p>
        )}
      </header>

      <nav className="ev-nav" aria-label="Sections">
        <div className="chips scroll">
          {SECTIONS.map(([id, label]) => <a key={id} className="chip" href={`#${id}`} onClick={(e) => { e.preventDefault(); document.getElementById(id)?.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); }}>{label}</a>)}
        </div>
      </nav>

      {hy && tr && (
        <Sec id="quality" n={++n} kicker="Quality"
          title={nScores ? `The hybrid is best on ${wins.length} of ${nScores} quality scores` : "Quality on the test scenes"}
          sub={<>{wins.length > 0 && <>Best on {wins.join(", ")}. </>}{losses.length > 0 && <>{losses.join("; ")}.</>} Mean over {b.dataset?.n_images ?? "all"} test images.</>}>
          <div className="ev-grid2">
            <div className="card">
              <h3 className="ev-h3"><Term k="psnr">PSNR</Term> <span className="muted">dB, higher is better</span></h3>
              <HBars unit="dB" rows={(b.quality ?? []).map((x) => ({ key: x.pipeline, label: SERIES_NAME[x.pipeline] ?? x.pipeline, value: x.psnr, colour: x.pipeline }))} />
            </div>
            <div className="card">
              <h3 className="ev-h3"><Term k="de">Colour error ΔE</Term> <span className="muted">lower is better</span></h3>
              <HBars lowerIsBetter rows={(b.quality ?? []).map((x) => ({ key: x.pipeline, label: SERIES_NAME[x.pipeline] ?? x.pipeline, value: x.delta_e, colour: x.pipeline }))} />
            </div>
          </div>
          <Table>
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Pipeline</th><th className="num">PSNR ↑</th><th className="num">SSIM ↑</th><th className="num">ΔE ↓</th><th className="num">LPIPS ↓</th></tr></thead>
                <tbody>
                  {(b.quality ?? []).map((x) => (
                    <tr key={x.pipeline} className={x.pipeline === "hybrid" ? "hybrid" : undefined}>
                      <td><i className={"sw " + cls(x.pipeline)} />{SERIES_NAME[x.pipeline] ?? x.pipeline}</td>
                      <td className="num">{fixed(x.psnr, 2)}<span className="unit">dB</span></td>
                      <td className="num">{fixed(x.ssim, 4)}</td>
                      <td className="num">{fixed(x.delta_e, 2)}</td>
                      <td className="num">{x.lpips != null ? fixed(x.lpips, 4) : <span className="dash">—</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Table>
        </Sec>
      )}

      {b.robustness && b.robustness.length > 0 && (
        <Sec id="robustness" n={++n} kicker="Robustness"
          title="Make the input harder, and the hybrid degrades last"
          sub={<>Each chart adds one problem to the RAW before any pipeline runs. {b.meta?.robustness_note ? `${b.meta.robustness_note[0].toUpperCase()}${b.meta.robustness_note.slice(1)}.` : ""} Tap a chart to read the values.</>}>
          <div className="ev-grid3">
            {b.robustness.map((rb) => (
              <div className="card" key={rb.title}>
                <h3 className="ev-h3">{rb.title}</h3>
                <LineChart x={rb.x} xLabel={rb.x_label} series={rb.series} />
              </div>
            ))}
          </div>
        </Sec>
      )}

      {fp32 && int8 && (
        <Sec id="efficiency" n={++n} kicker="Efficiency"
          title={`INT8 makes the network ${(fp32.size_mb / int8.size_mb).toFixed(1)}× smaller${fp32.peak_mem_mb && int8.peak_mem_mb ? ` and ${Math.round(fp32.peak_mem_mb / int8.peak_mem_mb)}× lighter on memory` : ""}`}
          sub={<>{(int8.params / 1e6).toFixed(2)}M parameters, measured on one 1-megapixel RAW on a CPU ({b.machine ?? "ONNX Runtime"}).{int8Cost !== null && <> It costs {int8Cost.toFixed(1)} dB on the ablation set, so the live demo runs FP32 and INT8 is the on-phone option.</>}</>}>
          <div className="ev-tiles">
            {[
              { k: "Model size", a: fp32.size_mb, b: int8.size_mb, u: "MB", d: 2 },
              { k: "Latency per megapixel", a: fp32.latency_ms_1mp, b: int8.latency_ms_1mp, u: "ms", d: 0 },
              ...(fp32.peak_mem_mb && int8.peak_mem_mb ? [{ k: "Peak memory", a: fp32.peak_mem_mb, b: int8.peak_mem_mb, u: "MB", d: 0 }] : []),
            ].map((t) => (
              <div className="card ev-tile" key={t.k}>
                <span className="ev-tile-k">{t.k}</span>
                <span className="ev-tile-v"><span className="from num">{fixed(t.a, t.d)}</span><i>→</i><span className="num">{fixed(t.b, t.d)}</span><small>{t.u}</small></span>
                <span className="ev-tile-s">FP32 → <Term k="int8">INT8</Term></span>
              </div>
            ))}
          </div>
          {b.efficiency_sweep && b.efficiency_sweep.length > 0 && (
            <Table label="Speed and memory at other image sizes">
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>RAW size</th><th className="num">Network FP32</th><th className="num">INT8</th><th className="num">Traditional</th><th className="num">Hybrid</th><th className="num">Hybrid memory</th></tr></thead>
                  <tbody>
                    {b.efficiency_sweep.map((r) => (
                      <tr key={String(r.megapixels)}>
                        <td>{r.megapixels} MP <span className="unit">{r.size}</span></td>
                        <td className="num">{Math.round(Number(r.net_fp32_ms))}<span className="unit">ms</span></td>
                        <td className="num">{r.net_int8_ms != null ? <>{Math.round(Number(r.net_int8_ms))}<span className="unit">ms</span></> : <span className="dash">—</span>}</td>
                        <td className="num">{Math.round(Number(r.traditional_ms))}<span className="unit">ms</span></td>
                        <td className="num">{Math.round(Number(r.hybrid_ms))}<span className="unit">ms</span></td>
                        <td className="num">{Math.round(Number(r.hybrid_mem_mb))}<span className="unit">MB</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {b.tiling && (
                <p className="table-note">
                  Large images are tiled automatically to bound memory. Inside the image, tiles match a single pass to{" "}
                  {Math.min(...b.tiling.variants.filter((v) => v.variant !== "single pass").map((v) => v.interior_psnr ?? 0)).toFixed(0)} dB PSNR, so there are no seams ({b.tiling.image}).
                </p>
              )}
            </Table>
          )}
        </Sec>
      )}

      {adaptive && (
        <Sec id="router" n={++n} kicker="Router"
          title={always ? `Choosing per photo beats always running the network by ${(adaptive.psnr - always.psnr).toFixed(1)} dB` : "Does the router earn its keep?"}
          sub={<>{b.router!.n_images} test images × {b.router!.conditions.length} conditions ({b.router!.conditions.join(", ")}). The router decides from the RAW alone, with thresholds tuned on the validation scene only.</>}>
          <div className="card">
            <h3 className="ev-h3"><Term k="psnr">PSNR</Term> <span className="muted">dB, higher is better</span></h3>
            <HBars unit="dB" digits={2} rows={modes.map((m) => ({ key: m.mode, label: m.mode, value: m.psnr, colour: /adaptive/i.test(m.mode) ? "hybrid" : /never/i.test(m.mode) ? "traditional" : "learned" }))} />
          </div>
          <Table>
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Mode</th><th className="num">PSNR ↑</th><th className="num">ΔE ↓</th><th className="num">Mean time</th><th className="num">Through the network</th></tr></thead>
                <tbody>
                  {modes.map((m) => (
                    <tr key={m.mode} className={/adaptive/i.test(m.mode) ? "hybrid" : undefined}>
                      <td>{m.mode}</td>
                      <td className="num">{fixed(m.psnr, 2)}<span className="unit">dB</span></td>
                      <td className="num">{fixed(m.delta_e, 2)}</td>
                      <td className="num">{Math.round(m.latency_ms)}<span className="unit">ms</span></td>
                      <td className="num">{Math.round(m.network_share * 100)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="table-note">
              Paths chosen as captured:{" "}
              {Object.entries(b.router!.paths_by_iso_as_captured).map(([k, v]) => `${k}: ${Object.entries(v).map(([p, c]) => `${c} ${p.replace("_", " + ")}`).join(", ")}`).join(" · ")}
            </p>
          </Table>
        </Sec>
      )}

      {full && deltas.length > 0 && (
        <Sec id="ablation" n={++n} kicker="Ablation"
          title={`Remove one part at a time: ${top2.join(" and ").toLowerCase()} matter most`}
          sub={<>Change in PSNR against the full hybrid ({fixed(full.psnr, 2)} dB){b.meta ? `, ${b.meta.subset_n} test images × ${b.meta.ablation_conditions.length} conditions` : ""}. {tiny > 0 && <>{tiny} parts move PSNR by less than 0.1 dB.</>}</>}>
          <div className="card">
            <DeltaBars rows={deltas} />
          </div>
          <Table>
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Variant</th><th className="num">PSNR</th><th className="num">SSIM</th><th className="num">ΔE</th></tr></thead>
                <tbody>
                  {(b.ablation ?? []).map((a) => (
                    <tr key={a.variant} className={a === full ? "hybrid" : undefined}>
                      <td>{a.variant}</td>
                      <td className="num">{fixed(a.psnr, 2)}<span className="unit">dB</span></td>
                      <td className="num">{fixed(a.ssim, 4)}</td>
                      <td className="num">{fixed(a.delta_e, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Table>
        </Sec>
      )}

      {phones.length > 0 && (
        <Sec id="phones" n={++n} kicker="Per phone"
          title={phoneWins === phones.length ? `Best on all ${phones.length} phones` : `Best on ${phoneWins} of ${phones.length} phones`}
          sub="As captured, mean PSNR in dB and colour error ΔE per phone in the test scenes.">
          <div className="card">
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Phone</th><th className="num">Images</th><th className="num">Traditional</th><th className="num">Network only</th><th className="num">Hybrid</th></tr></thead>
                <tbody>
                  {phones.map((g) => {
                    const best = Math.max(g.traditional.psnr, g.learned.psnr, g.hybrid.psnr);
                    return (
                      <tr key={g.group}>
                        <td>{g.group}</td>
                        <td className="num">{g.n}</td>
                        {(["traditional", "learned", "hybrid"] as const).map((p) => (
                          <td key={p} className={"num" + (g[p].psnr === best ? " best" : "")}>{fixed(g[p].psnr, 1)} <span className="unit">/ {fixed(g[p].delta_e, 1)}</span></td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </Sec>
      )}

      {raws.length > 0 && (
        <Sec id="phone-raw" n={++n} kicker="Real phone RAW"
          title={`${passed} of ${raws.length} real phone DNGs decode and develop correctly`}
          sub="Files from phones outside the training data. Each strip: sensor input, traditional, hybrid.">
          <div className="ev-gallery">
            {raws.map((p) => (
              <figure key={p.file} className="card">
                {p.preview && <img src={`/api/results-files/${p.preview.replace(/^results\//, "")}`} alt={`${p.file}: sensor input, traditional and hybrid renders`} loading="lazy" />}
                <figcaption>
                  <b>{p.file.replace(/\.dng$/i, "")}</b>
                  <span className={"pill " + (p.passed ? "positive" : "negative")}><span className="dot" />{p.passed ? "passes" : "fails"}</span>
                  <span className="small muted">
                    {p.libraw && `${p.libraw.raw_type === "Stack" ? "linear DNG" : `${p.libraw.cfa} Bayer`} · black ${p.libraw.black_level_per_channel[0]} / white ${p.libraw.white_level}`}
                    {p.route && ` · ${p.route.wb} white balance`}
                    {p.checks && ` · within ${p.checks.colour_vs_camera_wb_deg.toFixed(1)}° of the camera's colour`}
                  </span>
                </figcaption>
              </figure>
            ))}
          </div>
        </Sec>
      )}

      {b.failures && b.failures.length > 0 && (
        <Sec id="failures" n={++n} kicker="Failure cases" title="Where it still fails" sub="The worst test images, shown as they are.">
          <div className="ev-gallery one">
            {b.failures.map((f) => (
              <figure key={f.title} className="card">
                {f.image_url && <img src={f.image_url} alt={`${f.title}: input, hybrid output and ground truth`} loading="lazy" />}
                <div className="strip-labels" aria-hidden="true"><span>Input</span><span>Hybrid</span><span>Ground truth</span></div>
                <figcaption>
                  <b>{f.title}</b>
                  <span className="small muted">{f.description}</span>
                </figcaption>
              </figure>
            ))}
          </div>
        </Sec>
      )}
    </main>
  );
}
