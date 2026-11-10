import type { Row } from "../types";
import { label, pct, signed } from "../util";

const CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"];

export function AccuracyView({ rows, ref0 }: { rows: Row[]; ref0: Row }) {
  const sorted = [...rows].sort((a, b) => (b.accuracy?.mAP ?? 0) - (a.accuracy?.mAP ?? 0));
  const n = ref0.accuracy?.per_class ?? {};
  return <section id="accuracy"><h2>Accuracy analysis</h2><h3>Ground-truth COCO metrics on {ref0.accuracy?.n_images} fixed COCO val2017 images</h3>
    <p className="lead">Accuracy is pycocotools AP against COCO annotations, not detector confidence. Per-class AP rests on few examples — classes marked <span className="warn">⚠</span> have fewer than 30 ground-truth boxes and are indicative only. Images without any annotation are not part of the set, so values can differ slightly from official full-val2017 scores.</p>
    <div className="panel tablewrap"><table><thead><tr><th>Configuration</th><th className="n">mAP@[.5:.95] %</th><th className="n">Δ vs ref (pt)</th><th className="n">AP50 %</th>
      {CLASSES.map((c) => <th key={c} className="n">{c}<br /><span className="dim" style={{ textTransform: "none" }}>n={n[c]?.n_annotations ?? "?"} {(n[c]?.n_annotations ?? 0) < 30 && <span className="warn">⚠</span>}</span></th>)}</tr></thead>
      <tbody>{sorted.map((r) => { const d = ((r.accuracy?.mAP ?? 0) - (ref0.accuracy?.mAP ?? 0)) * 100; return <tr key={r.id}>
        <td>{label(r)}</td><td className="n">{pct(r.accuracy?.mAP, 2)}</td>
        <td className={"n " + (r.id === ref0.id ? "dim" : d >= 0 ? "pos" : "neg")}>{r.id === ref0.id ? "ref" : signed(d, 2)}</td>
        <td className="n">{pct(r.accuracy?.AP50, 2)}</td>
        {CLASSES.map((c) => <td key={c} className="n">{pct(r.accuracy?.per_class[c]?.AP)}</td>)}</tr>; })}</tbody></table>
      <div className="dim" style={{ fontSize: 12, marginTop: 6 }}>Per-class values are AP@[.5:.95] in %. Identical images across all rows. Configurations differing only in thread count or graph-optimization level should score identically (small float-level differences possible).</div>
    </div></section>;
}
