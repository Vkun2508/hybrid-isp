import type { RouteMode, SimPreset } from "../api";
import { sameSim } from "../lab/useLab";
import { Term } from "./Term";

interface SliderProps {
  id: string; label: React.ReactNode; value: number; min: number; max: number; step: number; zeroAt?: number;
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
        <span className="num value">{display}</span>
      </div>
      <div className="range-wrap">
        <span className="track" />
        <span className="fill" style={{ left: `${Math.min(z, p)}%`, width: `${Math.abs(p - z)}%` }} />
        {zeroAt !== undefined && <span className="zero" style={{ left: `calc(${z}% - 1px)` }} />}
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

/** One-tap hard inputs (the judges' favourite way to stress the pipeline on a phone). */
export const PRESETS: { id: string; label: string; sim: SimPreset | null }[] = [
  { id: "as", label: "As captured", sim: null },
  { id: "night", label: "Night · −3 EV", sim: { noise: 0, exposure_ev: -3, wb_shift: 0 } },
  { id: "iso", label: "High ISO", sim: { noise: 0.7, exposure_ev: 0, wb_shift: 0 } },
  { id: "wb", label: "Wrong white balance", sim: { noise: 0, exposure_ev: 0, wb_shift: 0.8 } },
  { id: "all", label: "Everything at once", sim: { noise: 0.5, exposure_ev: -2, wb_shift: -0.6 } },
];

interface Props {
  sim: SimPreset;
  preset: SimPreset;
  onSim: (s: SimPreset) => void;
}

export function Simulator({ sim, preset, onSim }: Props) {
  const ev = sim.exposure_ev;
  const wb = sim.wb_shift;
  return (
    <div className="sim">
      <div className="chips" role="group" aria-label="Quick presets">
        {PRESETS.map((p) => {
          const target = p.sim ?? preset;
          return (
            <button key={p.id} className="chip" aria-pressed={sameSim(sim, target)} onClick={() => onSim(target)}>
              {p.label}
            </button>
          );
        })}
      </div>
      <div className="sliders">
        <Slider
          id="noise" label="Sensor noise" value={sim.noise} min={0} max={1} step={0.05}
          display={sim.noise === 0 ? "as captured" : `≈ ISO ${iso(sim.noise).toLocaleString("en-US")}`}
          ends={["Clean", "Very high ISO"]} onChange={(v) => onSim({ ...sim, noise: v })}
        />
        <Slider
          id="ev" label={<>Exposure (<Term k="ev">EV</Term>)</>} value={ev} min={-5} max={2} step={0.25} zeroAt={0}
          display={`${ev > 0 ? "+" : ev < 0 ? "−" : "±"}${Math.abs(ev).toFixed(2)} EV`}
          ends={["5 stops dark", "2 bright"]} onChange={(v) => onSim({ ...sim, exposure_ev: v })}
        />
        <Slider
          id="wb" label={<Term k="wb">White-balance error</Term>} value={wb} min={-1} max={1} step={0.05} zeroAt={0}
          display={wb === 0 ? "none" : `${Math.round(Math.abs(wb) * 45)}% ${wb > 0 ? "too warm" : "too cool"}`}
          ends={["Too cool", "Too warm"]} onChange={(v) => onSim({ ...sim, wb_shift: v })}
        />
      </div>
      <p className="sim-note">Applied to the RAW before any pipeline runs, so all three get the same hard input.</p>
    </div>
  );
}

export function RouterControl({ mode, onMode }: { mode: RouteMode; onMode: (m: RouteMode) => void }) {
  const opts: [RouteMode, string][] = [["auto", "Adaptive (default)"], ["learned", "Always network"], ["fast", "Never network"]];
  return (
    <div className="router-ctl">
      <span className="small muted">Override the router:</span>
      <div className="chips" role="group" aria-label="Router mode">
        {opts.map(([m, label]) => (
          <button key={m} className="chip" aria-pressed={mode === m} onClick={() => onMode(m)}>{label}</button>
        ))}
      </div>
    </div>
  );
}
