import type { Row } from "../types";
import { diffFactors, f, label, signed } from "../util";

export function Breakdown({ rows, ref0 }: { rows: Row[]; ref0: Row }) {
  const ra = ref0.accuracy?.mAP ?? 0;
  const enriched = rows.map((r) => ({ r, factors: diffFactors(r, ref0) })).filter((x) => x.factors.length > 0);
  const single = enriched.filter((x) => x.factors.length === 1);
  const multi = enriched.filter((x) => x.factors.length > 1);
  const groups = [...new Set(single.map((x) => x.factors[0]))];
  const maxAbs = Math.max(1, ...enriched.map((x) => Math.abs((x.r.latency_p50_ms / ref0.latency_p50_ms - 1) * 100)));
  const Row_ = ({ r }: { r: Row }) => {
    const dl = (r.latency_p50_ms / ref0.latency_p50_ms - 1) * 100;
    const dm = ((r.accuracy?.mAP ?? 0) - ra) * 100;
    const dt = (r.benchmark.throughput_fps_pipeline / ref0.benchmark.throughput_fps_pipeline - 1) * 100;
    const dmem = r.memory.peak_rss_mb - ref0.memory.peak_rss_mb;
    return <tr>
      <td>{label(r)}</td>
      <td className="n" style={{ minWidth: 150 }}>
        <span style={{ display: "inline-block", height: 8, width: `${(Math.abs(dl) / maxAbs) * 90}px`, background: dl <= 0 ? "#4fb477" : "#e5484d", marginRight: 6 }} />
        <span className={dl <= 0 ? "pos" : "neg"}>{signed(dl)}%</span></td>
      <td className="n">{f(r.latency_p50_ms)}</td>
      <td className={"n " + (dm >= 0 ? "pos" : "neg")}>{signed(dm, 2)}</td>
      <td className={"n " + (dt >= 0 ? "pos" : "neg")}>{signed(dt)}%</td>
      <td className="n">{signed(dmem, 0)} MB</td></tr>;
  };
  const head = <thead><tr><th>Configuration</th><th className="n">Δ p50 latency</th><th className="n">p50 ms</th><th className="n">Δ mAP (pt)</th><th className="n">Δ throughput</th><th className="n">Δ peak RSS</th></tr></thead>;
  return <section id="breakdown"><h2>Optimization breakdown</h2><h3>One lever at a time, relative to {label(ref0)}</h3>
    <p className="lead">Green = faster / more accurate than the reference; red = regression. Single-lever rows differ from the reference in exactly one setting. Negative latency change is an improvement.</p>
    {groups.map((g) => <div className="panel" key={g}><div className="mono warn" style={{ marginBottom: 6, textTransform: "uppercase", fontSize: 12 }}>{g}</div>
      <div className="tablewrap"><table>{head}<tbody>{single.filter((x) => x.factors[0] === g).map((x) => <Row_ key={x.r.id} r={x.r} />)}</tbody></table></div></div>)}
    {multi.length > 0 && <div className="panel"><div className="mono dim" style={{ marginBottom: 6, textTransform: "uppercase", fontSize: 12 }}>Combined changes (not attributable to a single lever)</div>
      <div className="tablewrap"><table>{head}<tbody>{multi.map((x) => <Row_ key={x.r.id} r={x.r} />)}</tbody></table></div></div>}
  </section>;
}
