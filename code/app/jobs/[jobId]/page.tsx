"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type JobStatus = {
  job_id: string;
  status: string;
  events_written: number;
  error: string | null;
  progress?: number;
  stage?: string;
  shift_id?: string;
  summary?: {
    frames_processed: number;
    frames_with_hazards: number;
    workers_tracked: number;
    hazard_counts: Record<string, number>;
    detected_classes: string[];
    worker_compliance: Record<string, { ppe_confirmed: string[]; ppe_missing: string[]; compliant: boolean }>;
  };
};

type Event = {
  event_id: string;
  worker_id: string;
  camera_source: string;
  violation_type: string | null;
  severity: string | null;
  video_timestamp: number;
  clip_path: string | null;
  metadata: {
    explanation?: string;
    recommendation?: string;
    scene_summary?: string;
    reasoning?: string;
    confidence?: string;
    ppe_per_worker?: unknown[];
    event_start_ts?: number;
    event_end_ts?: number;
    event_duration_s?: number;
    event_frame_count?: number;
    yolo_hazard_types?: string[];
    hazard_type?: string;
    description?: string;
    ppe_worn?: string[];
    ppe_missing?: string[];
    track_id?: number;
  } | null;
  created_at: string;
};

type Shift = {
  shift_id: string;
  worker_id: string;
  date: string;
  status: string;
  violation_count: number;
  compliant_count: number;
  events: Event[];
};

type JobResults = {
  summary: JobStatus["summary"];
  vlm_results: any[];
  posture_summary?: {
    events?: number;
    frames_sampled?: number;
    total_frames?: number;
  };
  posture_events?: any[];
  posture_vlm?: any[];
};

export default function JobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const [job, setJob] = useState<JobStatus | null>(null);
  const [shift, setShift] = useState<Shift | null>(null);
  const [results, setResults] = useState<JobResults | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [polling, setPolling] = useState<NodeJS.Timer | null>(null);

  useEffect(() => {
    async function loadOnce() {
      const res = await fetch(`${BASE}/jobs/${jobId}`);
      if (!res.ok) { setLoading(false); return; }
      const j: JobStatus = await res.json();
      setJob(j);
      if (j.status === "COMPLETED" && j.shift_id) {
        const sr = await fetch(`${BASE}/shifts/${j.shift_id}`);
        if (sr.ok) setShift(await sr.json());
        const rr = await fetch(`${BASE}/jobs/${jobId}/results`);
        if (rr.ok) setResults(await rr.json());
      }
      setLoading(false);
    }

    // initial fetch
    loadOnce();

    // poll until done
    const timer = setInterval(loadOnce, 2000);
    setPolling(timer);
    return () => {
      clearInterval(timer);
      setPolling(null);
    };
  }, [jobId]);

  if (loading) {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--text-muted)", fontSize: "12px", padding: "40px 0" }}>
        <span className="spinner" /> Loading report…
      </div>
    );
  }
  if (!job) {
    return (
      <div style={{ padding: "40px 0", fontSize: "12px", color: "var(--red)", fontFamily: "var(--font-mono)" }}>
        JOB NOT FOUND
      </div>
    );
  }

  const summary = job.summary;
  const resultSummary = results?.summary ?? summary;
  const progress = job.progress ?? (job.status === "COMPLETED" ? 1 : 0);
  const stage = job.stage ?? (job.status === "COMPLETED" ? "completed" : job.status.toLowerCase());

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>

      {/* Breadcrumb */}
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "14px" }}>
          <a href="/jobs" style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)", textDecoration: "none", letterSpacing: "0.08em" }}>
            ← JOBS
          </a>
          <span style={{ color: "var(--border-2)", fontSize: "12px" }}>/</span>
          <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-dim)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "280px" }}>
            {jobId}
          </span>
          <StatusBadge status={job.status} />
        </div>
        <div className="page-eyebrow">Incident Report</div>
        <h1 className="page-title">Job Analysis</h1>
        <p className="page-sub mono">{jobId.slice(0, 20)}…</p>
      </div>

      {/* Progress */}
      {job.status !== "COMPLETED" && job.status !== "FAILED" && (
        <div className="card" style={{ padding: "12px 14px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
            <span className="card-label">Processing</span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--text-muted)" }}>
              {Math.round(progress * 100)}% · {stage}
            </span>
          </div>
          <div style={{ position: "relative", width: "100%", height: "10px", background: "var(--surface-2)", borderRadius: "999px", overflow: "hidden", border: "1px solid var(--border)" }}>
            <div style={{
              position: "absolute",
              left: 0,
              top: 0,
              bottom: 0,
              width: `${Math.max(3, Math.round(progress * 100))}%`,
              background: "linear-gradient(90deg, var(--accent) 0%, var(--blue) 100%)",
              transition: "width 0.4s ease",
            }} />
          </div>
        </div>
      )}

      {/* Error */}
      {job.status === "FAILED" && (
        <div style={{ padding: "14px 16px", background: "var(--red-bg)", border: "1px solid var(--red-border)", borderRadius: "8px" }}>
          <div style={{ fontSize: "9px", fontFamily: "var(--font-mono)", letterSpacing: "0.12em", color: "var(--red)", marginBottom: "6px" }}>PIPELINE ERROR</div>
          <pre style={{ fontSize: "12px", color: "#fca5a5", fontFamily: "var(--font-mono)", margin: 0, whiteSpace: "pre-wrap" }}>{job.error}</pre>
        </div>
      )}

      {/* Posture Analysis */}
      {results?.posture_summary && (
        <div className="card">
          <div className="card-head">
            <span className="card-label">Posture Analysis</span>
            <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
              REBA-inspired ergonomic risk
            </span>
          </div>

          {/* Stats row */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1px", background: "var(--border)", borderRadius: "6px", overflow: "hidden", marginBottom: "16px" }}>
            {[
              { label: "Risk Frames", value: results.posture_summary.events ?? 0, color: (results.posture_summary.events ?? 0) > 0 ? "var(--red)" : "var(--emerald)" },
              { label: "Frames Sampled", value: results.posture_summary.frames_sampled ?? "—", color: "var(--text)" },
              { label: "Total Frames", value: results.posture_summary.total_frames ?? "—", color: "var(--text-muted)" },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ background: "var(--surface-2)", padding: "12px 16px" }}>
                <div style={{ fontSize: "9px", fontFamily: "var(--font-mono)", letterSpacing: "0.1em", color: "var(--text-dim)", marginBottom: "6px" }}>{label.toUpperCase()}</div>
                <div style={{ fontSize: "22px", fontWeight: 800, fontFamily: "var(--font-mono)", color, letterSpacing: "-0.02em", lineHeight: 1 }}>{value}</div>
              </div>
            ))}
          </div>

          {/* VLM hazards */}
          {results.posture_vlm && results.posture_vlm.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              <div style={{ fontSize: "9px", fontFamily: "var(--font-mono)", letterSpacing: "0.1em", color: "var(--text-dim)" }}>
                VLM FINDINGS · {results.posture_vlm.reduce((a: number, c: any) => a + (c.vlm?.hazards?.length ?? 0), 0)} total
              </div>
              {results.posture_vlm.flatMap((chunk: any, idx: number) =>
                (chunk.vlm?.hazards ?? []).map((hz: any, i: number) => {
                  const frame = hz.best_frame_path || chunk.event?.representative_frame_path;
                  const ts = chunk.event?.representative_timestamp ?? chunk.event?.start_ts ?? 0;
                  return (
                    <div key={`${idx}-${i}`} style={{
                      display: "grid",
                      gridTemplateColumns: frame ? "1fr 160px" : "1fr",
                      gap: "14px",
                      background: "var(--surface-2)",
                      border: "1px solid var(--border)",
                      borderRadius: "8px",
                      padding: "14px",
                      alignItems: "start",
                    }}>
                      <div>
                        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                          <span className="pill">{hz.violation_type?.replace(/_/g, " ")}</span>
                          <span className="pill-soft">{hz.severity}</span>
                          <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-dim)" }}>t={ts.toFixed(1)}s</span>
                        </div>
                        {hz.explanation && (
                          <p style={{ fontSize: "12px", color: "var(--text-muted)", margin: "0 0 8px", lineHeight: 1.6 }}>{hz.explanation}</p>
                        )}
                        {hz.recommendation && (
                          <div style={{ fontSize: "11px", color: "var(--accent)", fontFamily: "var(--font-mono)", borderLeft: "2px solid var(--accent-dark)", paddingLeft: "8px" }}>
                            {hz.recommendation}
                          </div>
                        )}
                      </div>
                      {frame && (
                        <img
                          src={`${BASE}/jobs/${jobId}/frames/${frame}`}
                          alt="posture frame"
                          style={{ width: "100%", borderRadius: "6px", border: "1px solid var(--border)", display: "block" }}
                        />
                      )}
                    </div>
                  );
                })
              )}
            </div>
          )}
        </div>
      )}

      {/* Summary stats */}
      {resultSummary && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "10px" }}>
          <div className="stat-card" style={{ borderLeftColor: "var(--blue)" }}>
            <div className="stat-label">Frames Processed</div>
            <div className="stat-value" style={{ fontSize: "28px" }}>{resultSummary.frames_processed}</div>
          </div>
          <div className="stat-card" style={{ borderLeftColor: "var(--amber)" }}>
            <div className="stat-label">Hazard Frames</div>
            <div className="stat-value" style={{ fontSize: "28px", color: "var(--amber)" }}>{resultSummary.frames_with_hazards}</div>
          </div>
          <div className="stat-card" style={{ borderLeftColor: "var(--accent)" }}>
            <div className="stat-label">Workers Tracked</div>
            <div className="stat-value" style={{ fontSize: "28px" }}>{resultSummary.workers_tracked}</div>
          </div>
          <div className="stat-card" style={{ borderLeftColor: "var(--emerald)" }}>
            <div className="stat-label">Events Written</div>
            <div className="stat-value" style={{ fontSize: "28px" }}>{job.events_written}</div>
          </div>
        </div>
      )}

      {/* Hazard breakdown */}
      {resultSummary && Object.keys(resultSummary.hazard_counts).length > 0 && (
        <div className="card">
          <div className="card-head">
            <span className="card-label">Hazard Breakdown</span>
            <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
              {Object.values(resultSummary.hazard_counts).reduce((a, b) => a + b, 0)} total detections
            </span>
          </div>
          <div className="card-body" style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {Object.entries(resultSummary.hazard_counts).map(([k, v]) => (
              <div key={k} style={{
                display: "flex",
                alignItems: "center",
                gap: "10px",
                padding: "8px 14px",
                background: "var(--surface-2)",
                border: "1px solid var(--border)",
                borderRadius: "6px",
              }}>
                <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>{k.replace(/_/g, " ")}</span>
                <span style={{ fontSize: "16px", fontWeight: 800, fontFamily: "var(--font-mono)", color: "var(--amber)", letterSpacing: "-0.02em" }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Worker PPE compliance */}
      {resultSummary && Object.keys(resultSummary.worker_compliance ?? {}).length > 0 && (
        <div className="card">
          <div className="card-head">
            <span className="card-label">PPE Compliance</span>
          </div>
          <div>
            {Object.entries(resultSummary.worker_compliance).map(([tid, c], i, arr) => (
              <div key={tid} style={{
                display: "flex",
                alignItems: "center",
                gap: "16px",
                padding: "10px 16px",
                borderBottom: i < arr.length - 1 ? "1px solid var(--border)" : "none",
              }}>
                <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)", width: "72px" }}>Worker #{tid}</span>
                <span style={{
                  fontSize: "9px",
                  fontWeight: 700,
                  fontFamily: "var(--font-mono)",
                  letterSpacing: "0.1em",
                  color: c.compliant ? "var(--emerald)" : "var(--red)",
                  width: "72px",
                }}>
                  {c.compliant ? "✓ COMPLIANT" : "✗ VIOLATION"}
                </span>
                <div style={{ display: "flex", gap: "4px", flexWrap: "wrap", flex: 1 }}>
                  {c.ppe_confirmed.map(p => (
                    <span key={p} style={{ fontSize: "9px", background: "var(--emerald-bg)", color: "var(--emerald)", border: "1px solid var(--emerald-border)", borderRadius: "4px", padding: "2px 7px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>{p}</span>
                  ))}
                  {c.ppe_missing.map(p => (
                    <span key={p} style={{ fontSize: "9px", background: "var(--red-bg)", color: "var(--red)", border: "1px solid var(--red-border)", borderRadius: "4px", padding: "2px 7px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>no {p}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Safety Events */}
      {shift && (
        <div className="card">
          <div className="card-head">
            <span className="card-label">Safety Events</span>
            <span style={{ fontSize: "14px", fontWeight: 800, fontFamily: "var(--font-mono)", color: shift.events.length > 0 ? "var(--red)" : "var(--text)" }}>
              {shift.events.length}
            </span>
          </div>

          {shift.events.length === 0 ? (
            <div className="card-body" style={{ fontSize: "12px", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>
              No events recorded.
            </div>
          ) : (
            <>
              {/* Table header */}
              <div
                className="data-table-head"
                style={{ gridTemplateColumns: "64px 84px 1fr 80px 130px 18px" }}
              >
                {["Time", "Severity", "Violation", "Worker", "Range", ""].map(h => (
                  <span key={h} className="th">{h}</span>
                ))}
              </div>

              <div>
                {shift.events.map((e) => (
                  <div key={e.event_id} style={{ borderBottom: "1px solid var(--border)" }}>
                    {/* Row */}
                    <div
                      className="event-row"
                      style={{ gridTemplateColumns: "64px 84px 1fr 80px 130px 18px" }}
                      onClick={() => setExpanded(expanded === e.event_id ? null : e.event_id)}
                    >
                      <span style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                        {e.video_timestamp.toFixed(2)}s
                      </span>
                      <div><SeverityBadge severity={e.severity} /></div>
                      <span style={{ fontSize: "13px", color: "var(--text)", fontWeight: 500 }}>
                        {e.violation_type?.replace(/_/g, " ") ?? "—"}
                      </span>
                      <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                        {e.metadata?.track_id != null ? `#${e.metadata.track_id}` : "—"}
                      </span>
                      <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-dim)" }}>
                        {e.metadata?.event_start_ts != null
                          ? `${e.metadata.event_start_ts.toFixed(1)}s – ${e.metadata.event_end_ts?.toFixed(1)}s`
                          : "—"}
                      </span>
                      <span style={{ fontSize: "10px", color: "var(--text-dim)", textAlign: "right" }}>
                        {expanded === e.event_id ? "▲" : "▼"}
                      </span>
                    </div>

                    {/* Expanded detail */}
                    {expanded === e.event_id && (
                      <div className="event-detail animate-in">

                        {/* Frame image */}
                        {e.clip_path && (
                          <div>
                            <div style={{ fontSize: "9px", fontFamily: "var(--font-mono)", letterSpacing: "0.12em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "10px" }}>
                              Flagged Frame
                            </div>
                            <img
                              src={`${BASE}/jobs/${jobId}/frames/${e.clip_path}`}
                              alt="flagged frame"
                              style={{ borderRadius: "6px", maxHeight: "320px", border: "1px solid var(--border)", display: "block" }}
                            />
                          </div>
                        )}

                        {/* Explanation + recommendation */}
                        {e.metadata?.explanation && (
                          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                            <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "6px", padding: "14px" }}>
                              <div style={{ fontSize: "9px", fontFamily: "var(--font-mono)", letterSpacing: "0.12em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "8px" }}>Explanation</div>
                              <p style={{ fontSize: "13px", color: "var(--text)", margin: 0, lineHeight: 1.6 }}>{e.metadata.explanation}</p>
                            </div>
                            {e.metadata.recommendation && (
                              <div style={{ background: "var(--surface)", border: "1px solid var(--accent-dark)", borderRadius: "6px", padding: "14px" }}>
                                <div style={{ fontSize: "9px", fontFamily: "var(--font-mono)", letterSpacing: "0.12em", color: "var(--accent)", textTransform: "uppercase", marginBottom: "8px" }}>Recommendation</div>
                                <p style={{ fontSize: "13px", color: "var(--text)", margin: 0, lineHeight: 1.6 }}>{e.metadata.recommendation}</p>
                              </div>
                            )}
                          </div>
                        )}

                        {/* PPE worn / missing */}
                        {(e.metadata?.ppe_worn?.length || e.metadata?.ppe_missing?.length) ? (
                          <div>
                            <div style={{ fontSize: "9px", fontFamily: "var(--font-mono)", letterSpacing: "0.12em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "8px" }}>PPE Status</div>
                            <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                              {e.metadata?.ppe_worn?.map(p => (
                                <span key={p} style={{ fontSize: "9px", background: "var(--emerald-bg)", color: "var(--emerald)", border: "1px solid var(--emerald-border)", borderRadius: "4px", padding: "3px 8px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>{p}</span>
                              ))}
                              {e.metadata?.ppe_missing?.map(p => (
                                <span key={p} style={{ fontSize: "9px", background: "var(--red-bg)", color: "var(--red)", border: "1px solid var(--red-border)", borderRadius: "4px", padding: "3px 8px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>missing: {p}</span>
                              ))}
                            </div>
                          </div>
                        ) : null}

                        {/* YOLO triggers */}
                        {e.metadata?.yolo_hazard_types && e.metadata.yolo_hazard_types.length > 0 && (
                          <div>
                            <div style={{ fontSize: "9px", fontFamily: "var(--font-mono)", letterSpacing: "0.12em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "8px" }}>YOLO Triggers</div>
                            <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                              {e.metadata.yolo_hazard_types.map(h => (
                                <span key={h} style={{ fontSize: "10px", background: "var(--surface-3)", border: "1px solid var(--border)", borderRadius: "4px", padding: "3px 8px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>{h}</span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Raw metadata */}
                        <details>
                          <summary style={{ fontSize: "9px", fontFamily: "var(--font-mono)", letterSpacing: "0.1em", color: "var(--text-dim)", cursor: "pointer", userSelect: "none" }}>
                            RAW METADATA ▼
                          </summary>
                          <pre style={{ marginTop: "10px", fontSize: "11px", color: "var(--text-muted)", fontFamily: "var(--font-mono)", background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "6px", padding: "14px", overflow: "auto", lineHeight: 1.6 }}>
                            {JSON.stringify(e.metadata, null, 2)}
                          </pre>
                        </details>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* Detected classes */}
      {summary && summary.detected_classes.length > 0 && (
        <div className="card">
          <div className="card-head">
            <span className="card-label">Detected Classes</span>
            <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>{summary.detected_classes.length}</span>
          </div>
          <div className="card-body" style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
            {summary.detected_classes.map(c => (
              <span key={c} style={{
                fontSize: "10px",
                fontFamily: "var(--font-mono)",
                background: "var(--surface-2)",
                border: "1px solid var(--border)",
                borderRadius: "4px",
                padding: "3px 9px",
                color: "var(--text-muted)",
              }}>
                {c}
              </span>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const cls: Record<string, string> = {
    COMPLETED: "badge badge-completed",
    FAILED: "badge badge-failed",
    PROCESSING: "badge badge-processing",
  };
  return (
    <span className={cls[status] ?? "badge"}>
      {status === "PROCESSING" && (
        <span className="animate-pulse" style={{ display: "inline-block", width: "5px", height: "5px", borderRadius: "50%", background: "var(--amber)" }} />
      )}
      {status}
    </span>
  );
}

function SeverityBadge({ severity }: { severity: string | null }) {
  const cls: Record<string, string> = {
    CRITICAL: "badge badge-critical",
    HIGH:     "badge badge-high",
    MEDIUM:   "badge badge-medium",
    LOW:      "badge badge-low",
  };
  return (
    <span className={cls[severity ?? ""] ?? "badge badge-low"}>
      {severity ?? "—"}
    </span>
  );
}
