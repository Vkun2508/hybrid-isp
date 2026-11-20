import { useCallback, useEffect, useState } from "react";
import { ApiError, getBenchmarks, type Benchmarks } from "./api";

// Stale-while-revalidate: the last results show instantly on every visit, then refresh in the background
// (and again when the window regains focus), so switching tabs never flashes a skeleton.
let cached: Benchmarks | null = null;
let inflight: Promise<Benchmarks> | null = null;

export function useBenchmarks() {
  const [b, setB] = useState<Benchmarks | null>(cached);
  const [err, setErr] = useState<ApiError | null>(null);
  const load = useCallback(() => {
    setErr(null);
    inflight ??= getBenchmarks().finally(() => { inflight = null; });
    inflight
      .then((d) => { cached = d; setB(d); })
      .catch((e) => { if (!cached) setErr(e as ApiError); }); // keep showing stale data if a refresh fails
  }, []);
  useEffect(() => {
    load();
    window.addEventListener("focus", load);
    window.addEventListener("online", load);
    return () => { window.removeEventListener("focus", load); window.removeEventListener("online", load); };
  }, [load]);
  return { b, err, reload: load };
}

/** The headline numbers, computed from the evaluation file (never hard-coded). */
export function headline(b: Benchmarks | null) {
  const q = (p: string) => b?.quality?.find((x) => x.pipeline === p);
  const t = q("traditional"), h = q("hybrid"), l = q("learned");
  const int8 = b?.efficiency?.find((e) => /int8/i.test(e.variant));
  const fp32 = b?.efficiency?.find((e) => /fp32/i.test(e.variant));
  const under = b?.robustness?.find((r) => /expos/i.test(r.title));
  let dark: { ev: number; hybrid: number; others: number } | null = null;
  if (under) {
    const i = under.x.length - 1;
    const hy = under.series.find((s) => s.pipeline === "hybrid")?.psnr[i];
    const oth = under.series.filter((s) => s.pipeline !== "hybrid").map((s) => s.psnr[i]);
    if (hy !== undefined && oth.length) dark = { ev: under.x[i], hybrid: hy, others: Math.max(...oth) };
  }
  return {
    n: b?.dataset?.n_images ?? null,
    dPsnr: t && h ? h.psnr - t.psnr : null,
    dPsnrLearned: l && h ? h.psnr - l.psnr : null,
    deFrom: t?.delta_e ?? null, deTo: h?.delta_e ?? null,
    int8, fp32, dark,
  };
}
