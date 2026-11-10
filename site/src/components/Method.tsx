import type { Report, Row } from "../types";
import { f, label, pct } from "../util";

export function Method({ report, rows, ref0 }: { report: Report; rows: Row[]; ref0: Row }) {
  const env = report.env;
  const par = rows.filter((r) => r.parity);
  const agr = rows.filter((r) => r.agreement_vs_reference);
  const drift = report.drift_checks;
  const d0 = drift.find((d) => d.id.startsWith("drift-start")), d1 = drift.find((d) => d.id.startsWith("drift-end"));
  return <section id="method"><h2>Correctness & methodology</h2><h3>What was checked, and under which conditions</h3>
    <div className="grid2" style={{ gridTemplateColumns: "1fr 1fr" }}>
      <div className="panel"><div className="mono warn">FP32 numerical parity (ONNX Runtime vs PyTorch raw head output)</div>
        <div className="tablewrap"><table><thead><tr><th>Config</th><th className="n">max |Δ|</th><th className="n">mean |Δ|</th><th className="n">max rel</th><th>status</th></tr></thead>
          <tbody>{par.map((r) => <tr key={r.id}><td>{label(r)}</td><td className="n">{r.parity!.max_abs_err.toExponential(1)}</td><td className="n">{r.parity!.mean_abs_err.toExponential(1)}</td><td className="n">{r.parity!.max_rel_err.toExponential(1)}</td><td className={r.parity!.passed ? "pos" : "neg"}>{r.parity!.passed ? "pass" : "FAIL"}</td></tr>)}</tbody></table></div>
        {par[0] && <div className="dim" style={{ fontSize: 12 }}>Tolerance: atol {par[0].parity!.atol}, rtol {par[0].parity!.rtol} (outputs span ~1e-6 to ~4e2 in pixel units). {par[0].parity!.n_images} real images each. Relative error is taken only where |ref| &gt; 1e-3.</div>}
      </div>
      <div className="panel"><div className="mono warn">Detection agreement with reference (not accuracy)</div>
        <div className="tablewrap"><table><thead><tr><th>Config</th><th className="n">F1</th><th className="n">prec.</th><th className="n">recall</th></tr></thead>
          <tbody>{agr.map((r) => <tr key={r.id}><td>{label(r)}</td><td className="n">{pct(r.agreement_vs_reference!.f1)}</td><td className="n">{pct(r.agreement_vs_reference!.precision)}</td><td className="n">{pct(r.agreement_vs_reference!.recall)}</td></tr>)}</tbody></table></div>
        <div className="dim" style={{ fontSize: 12 }}>Same-class greedy matching at IoU ≥ {agr[0]?.agreement_vs_reference?.iou_thr}, score ≥ {agr[0]?.agreement_vs_reference?.score_thr}, against {label(ref0)}.</div>
      </div>
    </div>
    <div className="grid2" style={{ gridTemplateColumns: "1fr 1fr" }}>
      <div className="panel"><div className="mono warn">Hardware & software provenance</div>
        <table><tbody>
          {[["CPU", env.cpu_model], ["Cores", `${env.cpu_cores_physical} physical / ${env.cpu_cores_logical} logical`], ["RAM", `${env.ram_gb} GB unified`], ["OS", env.os], ["Python", env.python], ["PyTorch", env.torch], ["ONNX Runtime", env.onnxruntime], ["Execution", "CPU only (CPUExecutionProvider / PyTorch eager). No GPU, CUDA, MPS or CoreML."], ["Power source", env.power_source], ["Load avg (1 min) at start", String(env.load_avg_1m_at_start)], ["Git commit", env.git.commit ? `${env.git.commit.slice(0, 10)}${env.git.dirty ? " (dirty)" : ""}` : "not a git commit"], ["Run timestamp", env.timestamp_utc]].map(([k, v]) => <tr key={k}><td className="dim">{k}</td><td className="mono" style={{ fontSize: 12 }}>{v}</td></tr>)}
        </tbody></table></div>
      <div className="panel"><div className="mono warn">Benchmark & dataset settings</div>
        <table><tbody>
          {[["Mode", `${report.benchmark_mode}: ${ref0.benchmark.warmup} warmup, ${ref0.benchmark.n_per_repeat} timed × ${ref0.benchmark.repeats} repeats`], ["Primary metric", "p50 of model-forward latency (batch 1), median across repeats"], ["Pipeline metric", "preprocess + inference + decode/NMS"], ["Benchmark inputs", `${ref0.benchmark.n_distinct_images ?? 20} fixed val2017 images, cycled`], ["Eval set", `${report.dataset.n_images} COCO val2017 images`], ["Manifest", `${report.dataset.manifest} · sha256 ${report.dataset.manifest_sha256.slice(0, 16)}…`], ["Drift check (ref)", d0 && d1 ? `${f(d0.p50_inference_ms)} → ${f(d1.p50_inference_ms)} ms (${f(((d1.p50_inference_ms - d0.p50_inference_ms) / d0.p50_inference_ms) * 100, 1)}%) start→end` : "n/a"]].map(([k, v]) => <tr key={k}><td className="dim">{k}</td><td className="mono" style={{ fontSize: 12 }}>{v}</td></tr>)}
        </tbody></table></div>
    </div>
    {report.graph_opt_stats && <div className="panel"><div className="mono warn">What ONNX Runtime graph optimization changed (measured on the exported graph)</div>
      <p className="dim" style={{ fontSize: 13 }}>Graph nodes: {report.graph_opt_stats.total_nodes_original} original → {report.graph_opt_stats.total_nodes_basic} (basic) → {report.graph_opt_stats.total_nodes_all} (all). {report.graph_opt_stats.levels.all.QuickGelu ? `Level “all” fuses ${report.graph_opt_stats.levels.all.QuickGelu} Sigmoid→Mul pairs (the SiLU activation) into single QuickGelu ops and fuses Conv+activation/Add into ${report.graph_opt_stats.levels.all.FusedConv ?? 0} FusedConv nodes.` : ""}</p>
      <div className="tablewrap"><table><thead><tr><th>op</th><th className="n">original</th><th className="n">basic</th><th className="n">all</th></tr></thead><tbody>
        {[...new Set(Object.values(report.graph_opt_stats.levels).flatMap((l) => Object.keys(l)))].sort((a, b) => (report.graph_opt_stats!.levels.original[b] ?? 0) - (report.graph_opt_stats!.levels.original[a] ?? 0)).slice(0, 10).map((op) => <tr key={op}><td>{op}</td>{["original", "basic", "all"].map((l) => <td key={l} className="n">{report.graph_opt_stats!.levels[l][op] ?? 0}</td>)}</tr>)}</tbody></table></div></div>}
    <div className="panel"><div className="mono warn">Reproduce</div>
      <pre>{`uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]" && uv pip install --no-deps --no-build-isolation yolox
edgeforge prepare                    # weights, annotations, manifests, images
edgeforge sweep --mode rigorous      # runs every config, one fresh process each
edgeforge report --run-id <run-id>   # writes site/public/data/results.json`}</pre></div>
    <div className="panel"><div className="mono warn">Known limitations</div>
      <ul className="dim" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
        <li>{report.dataset.n_images} COCO val2017 images (those with at least one annotation); per-class AP for rare classes (see n) is noisy.</li>
        <li>Single laptop, shared with other applications and not thermally controlled; absolute latencies are machine- and load-specific. See the drift check above.</li>
        <li>Peak RSS is process-level (<code>ru_maxrss</code>) and includes interpreter and library overhead; PyTorch and ORT baselines differ partly for that reason.</li>
        <li>INT8 uses one calibration strategy (min-max, 100 images disjoint from evaluation). CoreML/MPS and models larger than Tiny are not part of these results. Latencies show a bootstrap 95% CI of the p50 over timed samples; mAP has no CI.</li>
        <li>Research prototype; not validated for automotive or safety-critical use.</li></ul></div>
  </section>;
}
