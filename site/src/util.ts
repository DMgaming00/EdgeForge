import type { Row } from "./types";

export const f = (v: number | null | undefined, d = 2) => (v == null ? "—" : v.toFixed(d));
export const pct = (v: number | null | undefined, d = 1) => (v == null ? "—" : (v * 100).toFixed(d));
export const signed = (v: number, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);

export const label = (r: Row) =>
  `${r.config.model === "yolox-tiny" ? "" : r.config.model.replace("yolox-", "").toUpperCase() + " · "}${r.config.backend === "pytorch" ? "PyTorch" : "ORT"} · ${r.config.resolution}px · t${r.config.threads}` +
  (r.config.backend === "onnxruntime" ? ` · opt:${r.config.graph_opt}` : "") + (r.config.precision === "int8" ? " · INT8" : "");

/** Factors in which `r` differs from `ref`. */
export function diffFactors(r: Row, ref: Row): string[] {
  const out: string[] = [];
  if (r.config.backend !== ref.config.backend) out.push("runtime");
  if (r.config.resolution !== ref.config.resolution) out.push("resolution");
  if (r.config.threads !== ref.config.threads) out.push("threads");
  if (r.config.graph_opt !== ref.config.graph_opt && r.config.backend === "onnxruntime" && ref.config.backend === "onnxruntime") out.push("graph optimization");
  if (r.config.model !== ref.config.model) out.push("model size");
  if (r.config.precision !== ref.config.precision) out.push("precision (INT8)");
  return out;
}

export function histogram(samples: number[], bins = 30) {
  const lo = Math.min(...samples), hi = Math.max(...samples);
  const w = (hi - lo) / bins || 1;
  const counts = new Array(bins).fill(0);
  for (const s of samples) counts[Math.min(bins - 1, Math.floor((s - lo) / w))]++;
  return { lo, w, counts };
}
