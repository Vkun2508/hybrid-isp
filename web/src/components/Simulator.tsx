import type { RouteMode, SimPreset } from "../api";

interface SliderProps {
  id: string; label: string; value: number; min: number; max: number; step: number; zeroAt?: number;
  display: string; ends: [string, string]; onChange: (v: number) => void;
}

function Slider({ id, label, value, min, max, step, zeroAt, display, ends, onChange }: SliderProps) {
  const pct = (v: number) => ((v - min) / (max - min)) * 100;
  const z = zeroAt === undefined ? 0 : pct(zeroAt);
  const p = pct(value);
  return (
    <div className="slider">
      <div className="slider-top">
        <label htmlFor={id}>{label}</label>
        <span className="num secondary">{display}</span>
      </div>
      <div className="range-wrap">
        <span className="track" />
        <span className="fill" style={{ left: `${Math.min(z, p)}%`, width: `${Math.abs(p - z)}%` }} />
        {zeroAt !== undefined && <span className="zero" style={{ left: `${z}%` }} />}
        <input
          id={id} className="range" type="range" min={min} max={max} step={step} value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          aria-valuetext={display}
        />
      </div>
      <div className="slider-ends"><span>{ends[0]}</span><span>{ends[1]}</span></div>
    </div>
  );
}

const iso = (t: number) => (t > 0 ? Math.round(100 * 2 ** (t * 7)) : 100);

interface Props {
  sim: SimPreset;
  onSim: (s: SimPreset) => void;
  onReset: () => void;
  dirty: boolean;
}

export function Simulator({ sim, onSim, onReset, dirty }: Props) {
  const ev = sim.exposure_ev;
  const wb = sim.wb_shift;
  return (
    <section className="card o-sim" aria-labelledby="sim-h">
      <div className="card-head">
        <h2 id="sim-h">Sensor simulator</h2>
        <p>Make the input harder</p>
      </div>
      <Slider
        id="noise" label="Sensor noise" value={sim.noise} min={0} max={1} step={0.05}
        display={sim.noise === 0 ? "as captured" : `≈ ISO ${iso(sim.noise).toLocaleString("en-US")}`}
        ends={["Clean", "Very high ISO"]} onChange={(v) => onSim({ ...sim, noise: v })}
      />
      <Slider
        id="ev" label="Exposure" value={ev} min={-5} max={2} step={0.25} zeroAt={0}
        display={`${ev > 0 ? "+" : ev < 0 ? "−" : "±"}${Math.abs(ev).toFixed(2)} EV`}
        ends={["5 stops under", "2 over"]} onChange={(v) => onSim({ ...sim, exposure_ev: v })}
      />
      <Slider
        id="wb" label="White-balance error" value={wb} min={-1} max={1} step={0.05} zeroAt={0}
        display={wb === 0 ? "none" : `${Math.round(Math.abs(wb) * 45)}% ${wb > 0 ? "too warm" : "too cool"}`}
        ends={["Too cool", "Too warm"]} onChange={(v) => onSim({ ...sim, wb_shift: v })}
      />
      <div className="sim-foot">
        <span className="small muted">Applied to the RAW before any pipeline runs.</span>
        <button className="btn text small" onClick={onReset} disabled={!dirty}>Reset to preset</button>
      </div>
    </section>
  );
}

export function RouterControl({ mode, onMode }: { mode: RouteMode; onMode: (m: RouteMode) => void }) {
  const opts: [RouteMode, string][] = [["auto", "Adaptive"], ["learned", "Always network"], ["fast", "Never network"]];
  return (
    <section className="card o-router" aria-labelledby="router-h">
      <div className="card-head">
        <h2 id="router-h">Hybrid router</h2>
        <p>How much network each photo gets</p>
      </div>
      <div className="seg" role="group" aria-label="Router mode">
        {opts.map(([m, label]) => (
          <button key={m} aria-pressed={mode === m} onClick={() => onMode(m)}>{label}</button>
        ))}
      </div>
    </section>
  );
}
