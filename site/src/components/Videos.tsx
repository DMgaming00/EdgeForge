import type { VideoEntry } from "../types";
import { f } from "../util";

export function Videos({ videos }: { videos: VideoEntry[] }) {
  return <section id="video"><h2>Perception demo</h2><h3>Annotated clips from the same inference backends</h3>
    {videos.length === 0 && <p className="dim">No demo clips have been generated yet. Run <code>edgeforge video</code>.</p>}
    {videos.map((v) => <div className="panel" key={v.file}>
      <div className="mono warn">{v.title}</div>
      <video src={v.file} controls muted loop playsInline preload="metadata" aria-label={v.title} />
      <table><thead><tr><th>Config</th><th className="n">inference p50</th><th className="n">offline fps</th><th className="n">detections (≥ threshold)</th></tr></thead>
        <tbody>{v.summary.runs.map((r) => <tr key={r.config.id}><td>{r.config.id} · {r.provider}</td><td className="n">{f(r.inference_ms_p50)} ms</td><td className="n">{f(r.offline_fps, 1)}</td><td className="n">{r.detections_total}</td></tr>)}</tbody></table>
      <div className="dim" style={{ fontSize: 12 }}>{v.summary.frames} frames. Overlay values are measured during the recorded run. Offline processing FPS (decode + inference + drawing) is not the playback rate of this clip and not a live-camera throughput claim. Source: {v.source}</div>
    </div>)}
  </section>;
}
