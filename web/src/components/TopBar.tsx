import type { ModelInfo } from "../api";

export type Tab = "overview" | "lab" | "results";

/** The product's mark: one 2x2 tile of the sensor's RGGB colour filter. */
export function Mark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 22 22" aria-hidden="true">
      <rect x="1" y="1" width="9" height="9" rx="2" fill="#e5566f" />
      <rect x="12" y="1" width="9" height="9" rx="2" fill="#44a282" />
      <rect x="1" y="12" width="9" height="9" rx="2" fill="#44a282" />
      <rect x="12" y="12" width="9" height="9" rx="2" fill="#4a8ad6" />
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
        <span className="dot" /> Placeholder model
      </span>
    );
  const params = model.params ? ` · ${(model.params / 1e6).toFixed(2)}M params` : "";
  const fmt = model.file?.includes("int8") ? " · INT8" : model.kind === "onnx" ? " · FP32" : "";
  return (
    <span className="pill positive model-pill" title={`${model.name}${model.file ? ` (${model.file})` : ""}`}>
      <span className="dot" /> Network live{fmt}{params}
    </span>
  );
}

const ICONS: Record<Tab, JSX.Element> = {
  overview: <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true"><path d="M3.5 10L11 4l7.5 6v8a1 1 0 0 1-1 1h-4v-5h-5v5h-4a1 1 0 0 1-1-1z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" /></svg>,
  lab: <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true"><rect x="3" y="5" width="16" height="12" rx="2" fill="none" stroke="currentColor" strokeWidth="1.6" /><path d="M11 5v12" stroke="currentColor" strokeWidth="1.6" /><circle cx="11" cy="11" r="2.2" fill="currentColor" /></svg>,
  results: <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true"><path d="M4 18V11M9 18V6M14 18v-5M19 18V8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>,
};
const TAB_NAME: Record<Tab, string> = { overview: "Overview", lab: "Try it", results: "Evidence" };

export function TopBar({ tab, onTab, model, onTour }: { tab: Tab; onTab: (t: Tab) => void; model: ModelInfo | null; onTour: () => void }) {
  return (
    <header className="topbar">
      <button className="brand" onClick={() => onTab("overview")} aria-label="Hybrid ISP, overview">
        <Mark />
        <span className="brand-name">Hybrid ISP</span>
      </button>
      <nav className="tabs" aria-label="Views">
        {(["overview", "lab", "results"] as Tab[]).map((t) => (
          <button key={t} className="tab" aria-current={tab === t ? "page" : undefined} onClick={() => onTab(t)}>
            {ICONS[t]}
            {TAB_NAME[t]}
          </button>
        ))}
      </nav>
      <div className="topbar-end">
        <ModelPill model={model} />
        <button className="btn primary demo-btn" onClick={onTour}>
          <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 3v10l8-5z" fill="currentColor" /></svg>
          Guided demo
        </button>
      </div>
    </header>
  );
}
