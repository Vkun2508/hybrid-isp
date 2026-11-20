import { useCallback, useEffect, useState } from "react";
import { flushSync } from "react-dom";
import { LabView } from "./components/Lab";
import { Overview } from "./components/Overview";
import { Results } from "./components/Results";
import { TopBar, type Tab } from "./components/TopBar";
import { Tour } from "./components/Tour";
import { useLab } from "./lab/useLab";
import { useBenchmarks } from "./useBenchmarks";

const fromHash = (): { tab: Tab; tour: boolean } => {
  const h = location.hash.replace("#", "");
  if (h === "tour") return { tab: "lab", tour: true };
  if (h === "lab" || h === "results") return { tab: h, tour: false };
  return { tab: "overview", tour: false };
};

export default function App() {
  const [tab, setTab] = useState<Tab>(() => fromHash().tab);
  const [tour, setTour] = useState(() => fromHash().tour);
  const lab = useLab();
  const bench = useBenchmarks();

  const changeTab = useCallback((t: Tab, opts?: { keepScroll?: boolean }) => {
    // push, so the phone's back button returns to the previous view instead of leaving the site
    const url = t === "overview" ? location.pathname : `#${t}`;
    if (fromHash().tab !== t || location.hash === "#tour") history.pushState(null, "", url);
    const apply = () => {
      setTab(t);
      if (t !== "lab") setTour(false); // the tour points at Lab elements; leaving the Lab ends it
      if (!opts?.keepScroll) window.scrollTo({ top: 0 });
    };
    const doc = document as Document & { startViewTransition?: (cb: () => void) => unknown };
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (doc.startViewTransition && !still) doc.startViewTransition(() => flushSync(apply));
    else apply();
  }, []);

  useEffect(() => {
    const on = () => {
      const h = fromHash();
      setTab(h.tab);
      if (h.tour) setTour(true);
      else if (h.tab !== "lab") setTour(false);
    };
    window.addEventListener("hashchange", on);
    window.addEventListener("popstate", on);
    return () => { window.removeEventListener("hashchange", on); window.removeEventListener("popstate", on); };
  }, []);

  const startTour = () => { if (tab !== "lab") changeTab("lab"); setTour(true); };

  return (
    <>
      <a className="sr-only" href="#main">Skip to content</a>
      <TopBar tab={tab} onTab={changeTab} model={lab.model} onTour={startTour} />
      {tab === "overview" && <Overview lab={lab} bench={bench.b} onTab={changeTab} onTour={startTour} />}
      {tab === "lab" && <LabView lab={lab} onTour={startTour} />}
      {tab === "results" && <Results bench={bench} />}
      {tour && (
        <Tour
          lab={lab} bench={bench.b}
          onExit={() => { setTour(false); if (location.hash === "#tour") history.replaceState(null, "", "#lab"); }}
          onEvidence={() => { setTour(false); changeTab("results"); }}
        />
      )}
      <footer className="foot">
        <p>
          Real RAW samples: SIDD-CC (Abdelhamed et al., CVPR 2018), from scenes held out of training. Photos: scikit-image test images (public domain).
        </p>
        <p>Uploads are processed in memory and never stored. Computer Vision Hackathon 2026 · Problem P2.</p>
      </footer>
    </>
  );
}
