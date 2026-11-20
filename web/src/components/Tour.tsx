import type { Benchmarks } from "../api";
import type { Lab } from "../lab/useLab";

export function Tour(_: { lab: Lab; bench: Benchmarks | null; onExit: () => void; onEvidence: () => void }) {
  return null;
}
