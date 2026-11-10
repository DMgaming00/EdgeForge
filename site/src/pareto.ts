export interface Pt { id: string; valid: boolean; latency: number | null | undefined; map: number | null | undefined }

const ok = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
export const eligible = (p: Pt) => p.valid && ok(p.latency) && ok(p.map);
export const dominates = (a: { latency: number; map: number }, b: { latency: number; map: number }) =>
  a.latency <= b.latency && a.map >= b.map && (a.latency < b.latency || a.map > b.map);

/** Minimise latency, maximise mAP. Invalid / incomplete points never participate. Ties are kept. */
export function paretoFront(points: Pt[]): string[] {
  const el = points.filter(eligible) as (Pt & { latency: number; map: number })[];
  return el.filter((p) => !el.some((q) => q !== p && dominates(q, p))).map((p) => p.id);
}
