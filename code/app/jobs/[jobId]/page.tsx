"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type JobStatus = {
  job_id: string;
  status: string;
  events_written: number;
  error: string | null;
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

export default function JobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const [job, setJob] = useState<JobStatus | null>(null);
  const [shift, setShift] = useState<Shift | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      const res = await fetch(`${BASE}/jobs/${jobId}`);
      if (!res.ok) { setLoading(false); return; }
      const j: JobStatus = await res.json();
      setJob(j);
      if (j.status === "COMPLETED" && j.shift_id) {
        const sr = await fetch(`${BASE}/shifts/${j.shift_id}`);
        if (sr.ok) setShift(await sr.json());
      }
      setLoading(false);
    }
    load();
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

      {/* Error */}
      {job.status === "FAILED" && (
        <div style={{ padding: "14px 16px", background: "var(--red-bg)", border: "1px solid var(--red-border)", borderRadius: "8px" }}>
          <div style={{ fontSize: "9px", fontFamily: "var(--font-mono)", letterSpacing: "0.12em", color: "var(--red)", marginBottom: "6px" }}>PIPELINE ERROR</div>
          <pre style={{ fontSize: "12px", color: "#fca5a5", fontFamily: "var(--font-mono)", margin: 0, whiteSpace: "pre-wrap" }}>{job.error}</pre>
        </div>
      )}

      {/* Summary stats */}
      {summary && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "10px" }}>
          <div className="stat-card" style={{ borderLeftColor: "var(--blue)" }}>
            <div className="stat-label">Frames Processed</div>
            <div className="stat-value" style={{ fontSize: "28px" }}>{summary.frames_processed}</div>
          </div>
          <div className="stat-card" style={{ borderLeftColor: "var(--amber)" }}>
            <div className="stat-label">Hazard Frames</div>
            <div className="stat-value" style={{ fontSize: "28px", color: "var(--amber)" }}>{summary.frames_with_hazards}</div>
          </div>
          <div className="stat-card" style={{ borderLeftColor: "var(--accent)" }}>
            <div className="stat-label">Workers Tracked</div>
            <div className="stat-value" style={{ fontSize: "28px" }}>{summary.workers_tracked}</div>
          </div>
          <div className="stat-card" style={{ borderLeftColor: "var(--emerald)" }}>
            <div className="stat-label">Events Written</div>
            <div className="stat-value" style={{ fontSize: "28px" }}>{job.events_written}</div>
          </div>
        </div>
      )}

      {/* Hazard breakdown */}
      {summary && Object.keys(summary.hazard_counts).length > 0 && (
        <div className="card">
          <div className="card-head">
            <span className="card-label">Hazard Breakdown</span>
            <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
              {Object.values(summary.hazard_counts).reduce((a, b) => a + b, 0)} total detections
            </span>
          </div>
          <div className="card-body" style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {Object.entries(summary.hazard_counts).map(([k, v]) => (
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
      {summary && Object.keys(summary.worker_compliance ?? {}).length > 0 && (
        <div className="card">
          <div className="card-head">
            <span className="card-label">PPE Compliance</span>
          </div>
          <div>
            {Object.entries(summary.worker_compliance).map(([tid, c], i, arr) => (
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
