import { useEffect, useMemo, useState } from "react";
import type { Report, Row, VideoEntry } from "./types";
import { paretoFront } from "./pareto";
import { Frontier } from "./components/Frontier";
import { Breakdown } from "./components/Breakdown";
import { Latency } from "./components/Latency";
import { AccuracyView } from "./components/AccuracyView";
import { Method } from "./components/Method";
import { Videos } from "./components/Videos";
import { f, label, pct, signed } from "./util";

export default function App() {
  const [report, setReport] = useState<Report | null>(null);
  const [videos, setVideos] = useState<VideoEntry[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [refId, setRefId] = useState<string>("");
  const [sel, setSel] = useState<string | null>(null);

  useEffect(() => {
    fetch("./data/results.json").then((r) => { if (!r.ok) throw new Error(`results.json: HTTP ${r.status}`); return r.json(); })
      .then((r: Report) => { setReport(r); setRefId(r.reference_id); setSel(r.reference_id); })
      .catch((e) => setErr(String(e)));
    fetch("./data/videos.json").then((r) => (r.ok ? r.json() : [])).then(setVideos).catch(() => setVideos([]));
  }, []);

  // Pareto membership is recomputed from the raw result rows in the browser.
  const rows: Row[] = useMemo(() => {
    if (!report) return [];
    const front = new Set(paretoFront(report.configs.map((c) => ({ id: c.id, valid: c.valid, latency: c.latency_p50_ms, map: c.accuracy?.mAP }))));
    return report.configs.map((c) => ({ ...c, pareto_optimal: front.has(c.id) }));
  }, [report]);

  if (err) return <div className="wrap"><h1>EdgeForge</h1><p className="neg">Could not load results: {err}. Run <code>edgeforge report --run-id …</code>.</p></div>;
  if (!report) return <div className="wrap"><p className="dim">Loading results…</p></div>;

  const ref = rows.find((r) => r.id === refId) ?? rows[0];
  const valid = rows.filter((r) => r.valid);
  const others = valid.filter((r) => r.id !== ref.id);
  const fastest = [...valid].sort((a, b) => a.latency_p50_ms - b.latency_p50_ms)[0];
  const env = report.env;
  const dRatio = ref.latency_p50_ms / fastest.latency_p50_ms;
  const dMap = ((fastest.accuracy?.mAP ?? 0) - (ref.accuracy?.mAP ?? 0)) * 100;
  const dd = report.drift_checks.length === 2 ? Math.abs(report.drift_checks[0].p50_inference_ms - report.drift_checks[1].p50_inference_ms) / Math.min(report.drift_checks[0].p50_inference_ms, report.drift_checks[1].p50_inference_ms) * 100 : null;
  const bestMap = [...valid].sort((a, b) => (b.accuracy?.mAP ?? 0) - (a.accuracy?.mAP ?? 0))[0];

  return (<>
    <nav><div className="wrap"><b>EDGEFORGE</b>
      {[["frontier", "Frontier"], ["breakdown", "Optimizations"], ["latency", "Latency"], ["accuracy", "Accuracy"], ["method", "Methodology"], ["video", "Video"]].map(([id, t]) => <a key={id} href={`#${id}`}>{t}</a>)}
    </div></nav>
    <div className="wrap">
      <header style={{ padding: "48px 0 8px" }}>
        <h2>Hardware-aware inference optimization</h2>
        <h1>How much accuracy does faster object detection really cost?</h1>
        <p className="lead">Measured accuracy (COCO val2017 subset, ground-truth mAP) against measured CPU latency for {valid.length} validated configurations of {ref.config.model}. Every number on this page is read from recorded experiment files.</p>
        <div className="kpis">
          <div className="kpi"><div className="l">CPU</div><div className="v" style={{ fontSize: 17 }}>{env.cpu_model}</div><div className="s">{env.cpu_cores_physical} cores · {env.ram_gb} GB · CPU execution only</div></div>
          <div className="kpi"><div className="l">Model</div><div className="v" style={{ fontSize: 17 }}>{ref.config.model}</div><div className="s">FP32 · batch 1 · pretrained</div></div>
          <div className="kpi"><div className="l">Valid configs</div><div className="v">{valid.length}<span className="dim"> / {rows.length}</span></div><div className="s">{rows.filter((r) => r.pareto_optimal).length} Pareto-optimal</div></div>
          <div className="kpi"><div className="l">Eval set</div><div className="v">{report.dataset.n_images}</div><div className="s">COCO val2017 images · {report.benchmark_mode} benchmark</div></div>
        </div>
        {dd !== null && dd > 10 && <div className="finding" style={{ borderColor: "var(--bad)" }}><b className="neg">Measurement noise warning. </b>The reference configuration's p50 changed by {f(dd, 0)}% between the start and end of this sweep (background load on the host: load avg {f(env.load_avg_1m_at_start, 0)} at start). Latency differences smaller than this drift, and rankings between configurations, are <b>not reliable</b> in this run. Re-run the sweep on an idle machine.</div>}
        <div className="finding">
          <b>Measured finding. </b>
          {fastest.id === ref.id
            ? <>The reference <span className="mono">{label(ref)}</span> is the fastest valid configuration at p50 {f(ref.latency_p50_ms)} ms.</>
            : <>Fastest valid configuration <span className="mono">{label(fastest)}</span>: model-only p50 {f(fastest.latency_p50_ms)} ms vs {f(ref.latency_p50_ms)} ms for the reference <span className="mono">{label(ref)}</span> ({f(dRatio)}× {dRatio >= 1 ? "faster" : "slower"}), with mAP {pct(fastest.accuracy?.mAP)} vs {pct(ref.accuracy?.mAP)} ({signed(dMap, 2)} points).</>}
          {bestMap && bestMap.id !== fastest.id && <> Highest mAP: <span className="mono">{label(bestMap)}</span> at {pct(bestMap.accuracy?.mAP)} and {f(bestMap.latency_p50_ms)} ms.</>}
          <div className="dim" style={{ fontSize: 12 }}>mAP on COCO val2017 images with ≥1 annotation ({report.dataset.n_images} images). Latency taken on a laptop (load avg at start {f(env.load_avg_1m_at_start)}, {env.power_source}); see Methodology.</div>
        </div>
      </header>
      <label className="dim" style={{ fontSize: 13 }}>Reference configuration:&nbsp;
        <select value={ref.id} onChange={(e) => setRefId(e.target.value)} aria-label="Reference configuration">
          {valid.map((r) => <option key={r.id} value={r.id}>{label(r)}</option>)}
        </select></label>
      <Frontier rows={rows} sel={sel} setSel={setSel} refId={ref.id} />
      <Breakdown rows={others} ref0={ref} />
      <Latency rows={rows} sel={sel ?? ref.id} setSel={setSel} />
      <AccuracyView rows={valid} ref0={ref} />
      <Method report={report} rows={rows} ref0={ref} />
      <Videos videos={videos} />
      <footer>EdgeForge · CPU-only research prototype. Not validated for automotive or safety use. Source results: {report.dataset.manifest} (sha256 {report.dataset.manifest_sha256.slice(0, 12)}…)</footer>
    </div>
  </>);
}
