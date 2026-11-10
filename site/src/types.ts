export interface Summary { mean: number; p50: number; p95: number; p99: number; min: number; max: number }
export type Stage = "preprocess" | "inference" | "postprocess" | "total";
export interface PerClass { AP: number | null; AP50: number | null; n_annotations: number }
export interface Accuracy {
  mAP: number; AP50: number; AP75: number; n_images: number; n_detections: number;
  per_class: Record<string, PerClass>;
}
export interface Parity {
  max_abs_err: number; mean_abs_err: number; max_rel_err: number; atol: number; rtol: number;
  passed: boolean; gating?: boolean; n_images: number; reference: string;
}
export interface Agreement { precision: number; recall: number; f1: number; matched: number; n_ref: number; n_test: number; iou_thr: number; score_thr: number }
export interface Cfg {
  id: string; model: string; backend: "pytorch" | "onnxruntime"; resolution: number;
  threads: number; graph_opt: "none" | "basic" | "all"; precision: string; batch_size: number;
}
export interface Row {
  id: string; config: Cfg; provider: string; valid: boolean; invalid_reasons: string[];
  latency_p50_ms: number; latency_p50_ci95?: [number, number]; repeat_spread_pct?: number | null; quantization?: { method: string; n_calibration: number; calibration_manifest: string; format: string } | null; pareto_optimal: boolean;
  benchmark: {
    mode: string; warmup: number; n_per_repeat: number; repeats: number;
    headline: Record<Stage, Summary>;
    throughput_fps_pipeline: number; throughput_fps_model_only: number;
    n_distinct_images?: number;
    runs: { samples_ms: Record<Stage, number[]>; summary: Record<Stage, Summary & { n: number }> }[];
  };
  memory: { rss_after_load_mb: number; peak_rss_mb: number };
  model_file: { size_mb: number; sha256: string; path: string };
  accuracy: Accuracy | null; parity: Parity | null; agreement_vs_reference: Agreement | null;
}
export interface Env {
  timestamp_utc: string; git: { commit: string | null; dirty: boolean | null }; os: string; cpu_model: string;
  cpu_cores_logical: number; cpu_cores_physical: number; ram_gb: number; python: string;
  torch: string; onnxruntime: string; power_source: string; load_avg_1m_at_start: number; notes: string;
}
export interface Report {
  schema_version: number; reference_id: string; env: Env; benchmark_mode: string;
  dataset: { manifest: string; manifest_sha256: string; n_images: number };
  pareto_ids: string[]; configs: Row[]; graph_opt_stats?: { total_nodes_original: number; total_nodes_basic: number; total_nodes_all: number; levels: Record<string, Record<string, number>> };
  drift_checks: { id: string; p50_inference_ms: number }[];
}
export interface VideoEntry { title: string; file: string; summary: { frames: number; runs: { config: Cfg; provider: string; inference_ms_p50: number; offline_fps: number; detections_total: number }[]; note: string }; source: string }
