import { useMemo, useRef, useState } from "react";
import type { Row } from "../types";
import { f, label, pct } from "../util";

const W = 760, H = 420, M = { l: 56, r: 16, t: 14, b: 44 };

export function Frontier({ rows, sel, setSel, refId }: { rows: Row[]; sel: string | null; setSel: (s: string) => void; refId: string }) {
  const [rt, setRt] = useState({ pytorch: true, onnxruntime: true });
  const [res, setRes] = useState<Record<number, boolean>>({});
  const [go, setGo] = useState({ none: true, basic: true, all: true });
  const [hover, setHover] = useState<string | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const resolutions = [...new Set(rows.map((r) => r.config.resolution))].sort((a, b) => a - b);

  const shown = rows.filter((r) => rt[r.config.backend] && (res[r.config.resolution] ?? true) && (r.config.backend === "pytorch" || go[r.config.graph_opt]));
  const plot = rows.filter((r) => r.accuracy);
  const xs = plot.map((r) => r.latency_p50_ms), ys = plot.map((r) => r.accuracy!.mAP);
  const x0 = Math.min(...xs) * 0.85, x1 = Math.max(...xs) * 1.08;
  const y0 = Math.min(...ys) - 0.01, y1 = Math.max(...ys) + 0.01;
  const X = (v: number) => M.l + ((v - x0) / (x1 - x0)) * (W - M.l - M.r);
  const Y = (v: number) => H - M.b - ((v - y0) / (y1 - y0)) * (H - M.t - M.b);
  const front = useMemo(() => rows.filter((r) => r.pareto_optimal && r.accuracy).sort((a, b) => a.latency_p50_ms - b.latency_p50_ms), [rows]);
  const path = front.map((r, i) => `${i ? "L" : "M"}${X(r.latency_p50_ms)},${Y(r.accuracy!.mAP)}`).join(" ");
  const tr = (n: number) => Array.from({ length: 5 }, (_, i) => n + ((i + 0) / 4));
  void tr;
  const hov = rows.find((r) => r.id === hover);
  const sr = rows.find((r) => r.id === sel);
  const ref = rows.find((r) => r.id === refId);

  return (
    <section id="frontier"><h2>Accuracy vs latency</h2><h3>Pareto frontier (minimise p50 inference latency, maximise mAP)</h3>
      <div className="filters">
        <fieldset><legend>Runtime</legend>{(["pytorch", "onnxruntime"] as const).map((k) => <label key={k}><input type="checkbox" checked={rt[k]} onChange={() => setRt({ ...rt, [k]: !rt[k] })} />{k === "pytorch" ? "PyTorch" : "ONNX Runtime"}</label>)}</fieldset>
        <fieldset><legend>Resolution</legend>{resolutions.map((v) => <label key={v}><input type="checkbox" checked={res[v] ?? true} onChange={() => setRes({ ...res, [v]: !(res[v] ?? true) })} />{v}</label>)}</fieldset>
        <fieldset><legend>ORT graph opt</legend>{(["none", "basic", "all"] as const).map((k) => <label key={k}><input type="checkbox" checked={go[k]} onChange={() => setGo({ ...go, [k]: !go[k] })} />{k}</label>)}</fieldset>
      </div>
      <div className="grid2">
        <div className="panel" ref={box} style={{ position: "relative", padding: 8 }}>
          <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Scatter plot of mAP versus p50 inference latency">
            {Array.from({ length: 6 }, (_, i) => { const v = y0 + ((y1 - y0) * i) / 5; return <g key={"y" + i}><line x1={M.l} x2={W - M.r} y1={Y(v)} y2={Y(v)} stroke="#23261f" /><text x={M.l - 8} y={Y(v) + 4} textAnchor="end">{pct(v)}</text></g>; })}
            {Array.from({ length: 6 }, (_, i) => { const v = x0 + ((x1 - x0) * i) / 5; return <g key={"x" + i}><line y1={M.t} y2={H - M.b} x1={X(v)} x2={X(v)} stroke="#23261f" /><text x={X(v)} y={H - M.b + 16} textAnchor="middle">{f(v, 0)}</text></g>; })}
            <text x={(W + M.l) / 2} y={H - 6} textAnchor="middle">p50 model-only inference latency (ms, CPU) — lower is better →</text>
            <text transform={`translate(14 ${H / 2}) rotate(-90)`} textAnchor="middle">COCO subset mAP@[.5:.95] (%)</text>
            <path d={path} fill="none" stroke="#f2a900" strokeWidth={1.5} strokeDasharray="4 3" />
            {rows.filter((r) => r.accuracy).map((r) => {
              const cx = X(r.latency_p50_ms), cy = Y(r.accuracy!.mAP);
              const on = shown.includes(r), isSel = r.id === sel;
              const fill = !r.valid ? "none" : r.pareto_optimal ? "#f2a900" : "#171915";
              const stroke = !r.valid ? "#e5484d" : r.pareto_optimal ? "#f2a900" : "#6fa8b8";
              const common = { fill, stroke, strokeWidth: isSel ? 3 : 1.5, opacity: on ? 1 : 0.12 } as const;
              return <g key={r.id} tabIndex={on ? 0 : -1} role="button" aria-label={`${label(r)}: ${f(r.latency_p50_ms)} ms, mAP ${pct(r.accuracy!.mAP)}`}
                style={{ cursor: "pointer", outline: "none" }} onMouseEnter={() => setHover(r.id)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(r.id)} onBlur={() => setHover(null)}
                onClick={() => setSel(r.id)} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && setSel(r.id)}>
                {r.config.backend === "pytorch" ? <rect x={cx - 6} y={cy - 6} width={12} height={12} {...common} /> : <circle cx={cx} cy={cy} r={7} {...common} />}
                {r.id === refId && <text x={cx + 10} y={cy - 8} style={{ fill: "#ece7dc" }}>REF</text>}
              </g>;
            })}
          </svg>
          {hov && hov.accuracy && <div className="tip" style={{ left: Math.min((X(hov.latency_p50_ms) / W) * 100, 60) + "%", top: (Y(hov.accuracy.mAP) / H) * 100 + "%" }}>
            <b>{label(hov)}</b><br />p50 {f(hov.latency_p50_ms)} ms · mAP {pct(hov.accuracy.mAP, 2)}<br />
            {hov.pareto_optimal ? <span className="warn">Pareto-optimal</span> : hov.valid ? "dominated" : <span className="neg">INVALID — excluded</span>}
          </div>}
          <div className="dim" style={{ fontSize: 12, padding: "4px 8px" }}>● ONNX Runtime ■ PyTorch · filled amber = Pareto-optimal · teal outline = dominated · red = failed validity checks. Frontier computed over all valid configurations regardless of filters.</div>
        </div>
        <div className="panel">
          {sr ? <>
            <h3 style={{ fontSize: 16 }}>{label(sr)}</h3>
            <span className={"tag " + (sr.valid ? "ok" : "bad")}>{sr.valid ? "valid" : "invalid"}</span>{" "}
            {sr.pareto_optimal && <span className="tag acc">pareto-optimal</span>}
            <table><tbody>
              <tr><td>p50 inference</td><td className="n">{f(sr.latency_p50_ms)} ms{sr.latency_p50_ci95 && <span className="dim"> [{f(sr.latency_p50_ci95[0])}–{f(sr.latency_p50_ci95[1])}]</span>}</td></tr>
              <tr><td>p95 / p99</td><td className="n">{f(sr.benchmark.headline.inference.p95)} / {f(sr.benchmark.headline.inference.p99)} ms</td></tr>
              <tr><td>pipeline p50</td><td className="n">{f(sr.benchmark.headline.total.p50)} ms</td></tr>
              <tr><td>throughput (pipeline)</td><td className="n">{f(sr.benchmark.throughput_fps_pipeline, 1)} fps</td></tr>
              <tr><td>mAP / AP50</td><td className="n">{pct(sr.accuracy?.mAP, 2)} / {pct(sr.accuracy?.AP50, 2)}</td></tr>
              <tr><td>peak RSS</td><td className="n">{f(sr.memory.peak_rss_mb, 0)} MB</td></tr>
              <tr><td>model file</td><td className="n">{f(sr.model_file.size_mb, 1)} MB</td></tr>
              <tr><td>provider</td><td className="n">{sr.provider}</td></tr>
              <tr><td>threads</td><td className="n">{sr.config.threads}</td></tr>
              {sr.parity && (sr.parity.gating === false ? <tr><td>quantization error vs FP32</td><td className="n">max |Δ| {sr.parity.max_abs_err.toFixed(2)} (not gated)</td></tr> : <tr><td>FP32 parity</td><td className={"n " + (sr.parity.passed ? "pos" : "neg")}>{sr.parity.passed ? "pass" : "FAIL"} (max |Δ| {sr.parity.max_abs_err.toExponential(1)})</td></tr>)}
              {ref && sr.id !== ref.id && <tr><td>vs reference</td><td className="n">{f(sr.latency_p50_ms - ref.latency_p50_ms)} ms · {f(((sr.accuracy?.mAP ?? 0) - (ref.accuracy?.mAP ?? 0)) * 100, 2)} pt</td></tr>}
            </tbody></table>
            {sr.invalid_reasons.length > 0 && <p className="neg">{sr.invalid_reasons.join("; ")}</p>}
          </> : <p className="dim">Click a point to inspect a configuration.</p>}
        </div>
      </div>
    </section>
  );
}
