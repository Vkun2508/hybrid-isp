import type { Benchmarks } from "../api";
import type { Lab } from "../lab/useLab";
import type { Tab } from "./TopBar";

export function Overview(_: { lab: Lab; bench: Benchmarks | null; onTab: (t: Tab) => void; onTour: () => void }) {
  return <main id="main" className="overview"><h1>Overview</h1></main>;
}
