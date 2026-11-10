import type { Row, Stage } from "../types";
import { f, histogram, label } from "../util";

const STAGE: [Stage, string, string][] = [["preprocess", "preprocess", "#6fa8b8"], ["inference", "inference", "#f2a900"], ["postprocess", "decode + NMS", "#8a8f84"]];

export function Latency({ rows, sel, setSel }: { rows: Row[]; sel: string; setSel: (s: string) => void }) {
  const r = rows.find((x) => x.id === sel) ?? rows[0];
  const samples = r.benchmark.runs.flatMap((x) => x.samples_ms.inference);
  const h = histogram(samples, 32);
  const mx = Math.max(...h.counts);
  const W = 640, H = 200, pad = 36;
  const bw = (W - pad - 8) / h.counts.length;
  const p = r.benchmark.headline;
  const tot = p.preprocess.p50 + p.inference.p50 + p.postprocess.p50;
  const sorted = [...rows].sort((a, b) => a.latency_p50_ms - b.latency_p50_ms);
  return <section id="latency"><h2>Latency analysis</h2><h3>Distribution and stage breakdown</h3>
    <label className="dim" style={{ fontSize: 13 }}>Configuration:&nbsp;<select value={r.id} onChange={(e) => setSel(e.target.value)}>{rows.map((x) => <option key={x.id} value={x.id}>{label(x)}</option>)}</select></label>
    <div className="grid2" style={{ marginTop: 12 }}>
      <div className="panel">
        <div className="dim" style={{ fontSize: 12 }}>Model-only inference latency, {samples.length} timed samples ({r.benchmark.repeats} repeats × {r.benchmark.n_per_repeat}; {r.benchmark.warmup} warmup passes discarded)</div>
        <svg viewBox={`0 0 ${W} ${H + 28}`} width="100%" role="img" aria-label="Latency histogram">
          {h.counts.map((c, i) => <rect key={i} x={pad + i * bw} y={H - (c / mx) * (H - 10)} width={bw - 1} height={(c / mx) * (H - 10)} fill="#6fa8b8" />)}
          {[["p50", p.inference.p50, "#f2a900"], ["p95", p.inference.p95, "#ece7dc"], ["p99", p.inference.p99, "#e5484d"]].map(([n, v, c]) => {
            const x = pad + (((v as number) - h.lo) / (h.w * h.counts.length)) * (W - pad - 8);
            return <g key={n as string}><line x1={x} x2={x} y1={0} y2={H} stroke={c as string} strokeDasharray="3 3" /><text x={x + 3} y={12} style={{ fill: c as string }}>{n as string} {f(v as number)}</text></g>;
          })}
          <text x={pad} y={H + 16}>{f(h.lo)} ms</text><text x={W - 8} y={H + 16} textAnchor="end">{f(h.lo + h.w * h.counts.length)} ms</text>
        </svg>
      </div>
      <div className="panel"><div className="dim" style={{ fontSize: 12, marginBottom: 6 }}>Pipeline p50 stage split ({f(tot)} ms sum of stage medians)</div>
        <div style={{ display: "flex", height: 18, border: "1px solid var(--border)" }}>
          {STAGE.map(([k, , c]) => <div key={k} title={k} style={{ width: `${(p[k].p50 / tot) * 100}%`, background: c }} />)}</div>
        <table><tbody>{STAGE.map(([k, n, c]) => <tr key={k}><td><span style={{ color: c }}>■</span> {n}</td><td className="n">{f(p[k].p50)} ms</td><td className="n dim">{f((p[k].p50 / tot) * 100, 0)}%</td></tr>)}
          <tr><td>pipeline total p50</td><td className="n">{f(p.total.p50)} ms</td><td /></tr>
          <tr><td>runtime / provider</td><td className="n" colSpan={2}>{r.provider}</td></tr>
          <tr><td>threads</td><td className="n" colSpan={2}>{r.config.threads}</td></tr></tbody></table>
      </div>
    </div>
    <div className="panel tablewrap"><table><thead><tr><th>Configuration</th><th className="n">p50</th><th className="n">p95</th><th className="n">p99</th><th className="n">min</th><th className="n">max</th><th className="n">fps (pipeline)</th><th className="n">fps (model only)</th></tr></thead>
      <tbody>{sorted.map((x) => <tr key={x.id} className={"click " + (x.id === r.id ? "sel" : "")} onClick={() => setSel(x.id)}>
        <td>{label(x)}</td>{(["p50", "p95", "p99", "min", "max"] as const).map((k) => <td key={k} className="n">{f(x.benchmark.headline.inference[k])}</td>)}
        <td className="n">{f(x.benchmark.throughput_fps_pipeline, 1)}</td><td className="n">{f(x.benchmark.throughput_fps_model_only, 1)}</td></tr>)}</tbody></table>
      <div className="dim" style={{ fontSize: 12, marginTop: 6 }}>Latencies in ms: model-only inference stage. p50/p95/p99 are medians across repeats of each repeat's percentile.</div></div>
  </section>;
}
