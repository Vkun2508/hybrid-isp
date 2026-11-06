import type { ModelInfo } from "../api";

export type Tab = "lab" | "results";

/** The product's mark: one 2x2 tile of the sensor's RGGB colour filter. */
export function Mark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 22 22" aria-hidden="true">
      <rect x="1" y="1" width="9" height="9" rx="1.5" fill="#e5566f" />
      <rect x="12" y="1" width="9" height="9" rx="1.5" fill="#44a282" />
      <rect x="1" y="12" width="9" height="9" rx="1.5" fill="#44a282" />
      <rect x="12" y="12" width="9" height="9" rx="1.5" fill="#4a8ad6" />
    </svg>
  );
}

export function ModelPill({ model }: { model: ModelInfo | null }) {
  if (!model) return <span className="pill model-pill">Loading model…</span>;
  if (model.load_error)
    return (
      <span className="pill negative model-pill" title={model.load_error}>
        <span className="dot" /> Model failed to load · classical stand-in
      </span>
    );
  if (model.is_stub)
    return (
      <span className="pill warning model-pill" title="The trained network is still training; a classical denoiser with the same interface stands in.">
        <span className="dot" /> Placeholder model · training in progress
      </span>
    );
  const params = model.params ? ` · ${(model.params / 1e6).toFixed(2)}M params` : "";
  const fmt = model.file?.includes("int8") ? " · INT8" : model.kind === "onnx" ? " · FP32" : "";
  return (
    <span className="pill positive model-pill" title={model.file ?? undefined}>
      <span className="dot" /> Learned core: {model.name.split(" (")[0]}{fmt}{params}
    </span>
  );
}

export function TopBar({ tab, onTab, model }: { tab: Tab; onTab: (t: Tab) => void; model: ModelInfo | null }) {
  return (
    <header className="topbar">
      <div className="brand">
        <Mark />
        <span className="brand-name">Hybrid ISP</span>
        <span className="brand-sub">Classical + learned camera pipeline</span>
      </div>
      <nav className="tabs" role="tablist" aria-label="Views">
        {(["lab", "results"] as Tab[]).map((t) => (
          <button key={t} role="tab" className="tab" aria-selected={tab === t} onClick={() => onTab(t)}>
            {t === "lab" ? "Lab" : "Results"}
          </button>
        ))}
      </nav>
      <div className="topbar-end">
        <ModelPill model={model} />
      </div>
    </header>
  );
}
