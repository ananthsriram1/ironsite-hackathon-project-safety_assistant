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

  if (loading) return <p className="text-zinc-500 text-sm p-8">Loading…</p>;
  if (!job) return <p className="text-red-400 text-sm p-8">Job not found.</p>;

  const summary = job.summary;

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">

      {/* Header */}
      <div className="flex items-center gap-3">
        <a href="/jobs" className="text-zinc-500 hover:text-white text-sm transition-colors">← Jobs</a>
        <span className="text-zinc-700">/</span>
        <p className="text-sm font-mono text-zinc-400 truncate">{jobId}</p>
        <StatusBadge status={job.status} />
      </div>

      {job.status === "FAILED" && (
        <div className="rounded-xl border border-red-900 bg-red-950/30 p-4">
          <p className="text-sm text-red-400 font-mono">{job.error}</p>
        </div>
      )}

      {/* Summary stats */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat label="Frames processed" value={summary.frames_processed} />
          <Stat label="Hazard frames" value={summary.frames_with_hazards} />
          <Stat label="Workers tracked" value={summary.workers_tracked} />
          <Stat label="Events written" value={job.events_written} />
        </div>
      )}

      {/* Hazard counts */}
      {summary && Object.keys(summary.hazard_counts).length > 0 && (
        <div className="rounded-xl border border-zinc-800 p-5">
          <h2 className="text-sm font-medium text-zinc-400 mb-3">Hazard Breakdown</h2>
          <div className="flex flex-wrap gap-2">
            {Object.entries(summary.hazard_counts).map(([k, v]) => (
              <span key={k} className="text-xs bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-1.5">
                <span className="text-zinc-400">{k.replace(/_/g, " ")}</span>
                <span className="text-white font-semibold ml-2">{v}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Worker compliance */}
      {summary && Object.keys(summary.worker_compliance ?? {}).length > 0 && (
        <div className="rounded-xl border border-zinc-800 p-5">
          <h2 className="text-sm font-medium text-zinc-400 mb-3">Worker PPE Compliance</h2>
          <div className="space-y-2">
            {Object.entries(summary.worker_compliance).map(([tid, c]) => (
              <div key={tid} className="flex items-center gap-4 py-2 border-b border-zinc-800/50 last:border-0">
                <span className="text-xs text-zinc-500 w-20">Worker #{tid}</span>
                <span className={`text-xs font-medium w-20 ${c.compliant ? "text-emerald-400" : "text-red-400"}`}>
                  {c.compliant ? "✓ Compliant" : "✗ Violation"}
                </span>
                <div className="flex gap-1 flex-wrap">
                  {c.ppe_confirmed.map(p => (
                    <span key={p} className="text-xs bg-emerald-950 text-emerald-400 border border-emerald-800 rounded px-2 py-0.5">{p}</span>
                  ))}
                  {c.ppe_missing.map(p => (
                    <span key={p} className="text-xs bg-red-950 text-red-400 border border-red-800 rounded px-2 py-0.5">no {p}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Events */}
      {shift && (
        <div className="rounded-xl border border-zinc-800 overflow-hidden">
          <div className="px-5 py-4 border-b border-zinc-800">
            <h2 className="text-sm font-medium text-zinc-400">
              Safety Events <span className="text-white ml-1">{shift.events.length}</span>
            </h2>
          </div>

          {shift.events.length === 0 ? (
            <p className="text-zinc-600 text-sm p-5">No events recorded.</p>
          ) : (
            <div className="divide-y divide-zinc-800">
              {shift.events.map((e) => (
                <div key={e.event_id}>
                  {/* Row */}
                  <div
                    onClick={() => setExpanded(expanded === e.event_id ? null : e.event_id)}
                    className="flex items-center gap-4 px-5 py-3 hover:bg-zinc-900 cursor-pointer transition-colors"
                  >
                    <span className="font-mono text-zinc-300 text-xs w-16">{e.video_timestamp.toFixed(2)}s</span>
                    <SeverityBadge severity={e.severity} />
                    <span className="text-sm text-white flex-1">{e.violation_type?.replace(/_/g, " ") ?? "—"}</span>
                    {e.metadata?.track_id != null && (
                      <span className="text-xs text-zinc-500">Worker #{e.metadata.track_id}</span>
                    )}
                    {e.metadata?.event_start_ts != null && (
                      <span className="text-xs text-zinc-600 font-mono">
                        {e.metadata.event_start_ts.toFixed(1)}s – {e.metadata.event_end_ts?.toFixed(1)}s
                      </span>
                    )}
                    <span className="text-zinc-600 text-xs">{expanded === e.event_id ? "▲" : "▼"}</span>
                  </div>

                  {/* Expanded */}
                  {expanded === e.event_id && (
                    <div className="bg-zinc-950 px-5 py-4 space-y-4">

                      {/* Frame image */}
                      {e.clip_path && (
                        <div>
                          <p className="text-xs text-zinc-500 mb-2">Flagged Frame</p>
                          <img
                            src={`${BASE}/jobs/${jobId}/frames/${e.clip_path}`}
                            alt="flagged frame"
                            className="rounded-lg max-h-80 border border-zinc-800"
                          />
                        </div>
                      )}

                      {/* Explanation + recommendation */}
                      {e.metadata?.explanation && (
                        <div className="grid grid-cols-2 gap-4">
                          <div className="rounded-lg bg-zinc-900 p-3">
                            <p className="text-xs text-zinc-500 mb-1">Explanation</p>
                            <p className="text-sm text-white">{e.metadata.explanation}</p>
                          </div>
                          {e.metadata.recommendation && (
                            <div className="rounded-lg bg-zinc-900 p-3">
                              <p className="text-xs text-zinc-500 mb-1">Recommendation</p>
                              <p className="text-sm text-white">{e.metadata.recommendation}</p>
                            </div>
                          )}
                        </div>
                      )}

                      {/* PPE worn / missing on this worker */}
                      {(e.metadata?.ppe_worn || e.metadata?.ppe_missing) && (
                        <div className="flex gap-2 flex-wrap">
                          {e.metadata.ppe_worn?.map(p => (
                            <span key={p} className="text-xs bg-emerald-950 text-emerald-400 border border-emerald-800 rounded px-2 py-0.5">{p}</span>
                          ))}
                          {e.metadata.ppe_missing?.map(p => (
                            <span key={p} className="text-xs bg-red-950 text-red-400 border border-red-800 rounded px-2 py-0.5">missing: {p}</span>
                          ))}
                        </div>
                      )}

                      {/* YOLO hazard types that triggered this event */}
                      {e.metadata?.yolo_hazard_types && e.metadata.yolo_hazard_types.length > 0 && (
                        <div>
                          <p className="text-xs text-zinc-500 mb-1">YOLO triggers</p>
                          <div className="flex gap-2 flex-wrap">
                            {e.metadata.yolo_hazard_types.map(h => (
                              <span key={h} className="text-xs bg-zinc-900 border border-zinc-700 rounded px-2 py-0.5 text-zinc-300">{h}</span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Raw metadata dump */}
                      <details className="group">
                        <summary className="text-xs text-zinc-600 cursor-pointer hover:text-zinc-400">Raw metadata</summary>
                        <pre className="mt-2 text-xs text-zinc-500 overflow-auto bg-zinc-900 rounded-lg p-3">
                          {JSON.stringify(e.metadata, null, 2)}
                        </pre>
                      </details>

                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Detected classes */}
      {summary && summary.detected_classes.length > 0 && (
        <div className="rounded-xl border border-zinc-800 p-5">
          <h2 className="text-sm font-medium text-zinc-400 mb-3">Detected Classes</h2>
          <div className="flex flex-wrap gap-2">
            {summary.detected_classes.map(c => (
              <span key={c} className="text-xs bg-zinc-900 border border-zinc-800 rounded px-2 py-1 text-zinc-400">{c}</span>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    COMPLETED: "bg-emerald-950 text-emerald-400 border-emerald-800",
    FAILED: "bg-red-950 text-red-400 border-red-800",
    PROCESSING: "bg-yellow-950 text-yellow-400 border-yellow-800",
  };
  return (
    <span className={`text-xs font-medium px-2.5 py-1 rounded-lg border ${colors[status] ?? "border-zinc-700 text-zinc-400"}`}>
      {status === "PROCESSING" && <span className="animate-pulse">● </span>}
      {status}
    </span>
  );
}

function SeverityBadge({ severity }: { severity: string | null }) {
  const colors: Record<string, string> = {
    CRITICAL: "bg-red-950 text-red-400 border-red-800",
    HIGH: "bg-orange-950 text-orange-400 border-orange-800",
    MEDIUM: "bg-yellow-950 text-yellow-400 border-yellow-800",
    LOW: "bg-zinc-900 text-zinc-400 border-zinc-700",
  };
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded border ${colors[severity ?? ""] ?? "border-zinc-700 text-zinc-500"}`}>
      {severity ?? "—"}
    </span>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
      <p className="text-xs text-zinc-500 mb-1">{label}</p>
      <p className="text-2xl font-semibold text-white">{value}</p>
    </div>
  );
}
