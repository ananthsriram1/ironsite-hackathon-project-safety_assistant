"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Event = {
  event_id: string;
  camera_source: string;
  event_category: string;
  violation_type: string | null;
  severity: string | null;
  video_timestamp: number;
  clip_path: string | null;
  metadata: Record<string, unknown> | null;
  created_at: string;
};

type JobDb = {
  job_id: string;
  shift: {
    id: string;
    worker_id: string;
    site_id: string;
    date: string;
    status: string;
    started_at: string;
    ended_at: string | null;
    violation_count: number;
    compliant_count: number;
  };
  worker: { id: string; display_name: string | null; total_violations: number; last_seen_at: string | null } | null;
  events: Event[];
  total_events: number;
};

type JobStatus = {
  job_id: string;
  status: string;
  events_written: number;
  error: string | null;
  summary?: Record<string, unknown>;
};

export default function JobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [dbData, setDbData] = useState<JobDb | null>(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      const res = await fetch(`${BASE}/jobs/${jobId}`);
      if (!res.ok) return;
      const s: JobStatus = await res.json();
      setStatus(s);
      if (s.status === "COMPLETED") {
        const dbRes = await fetch(`${BASE}/jobs/${jobId}/db`);
        if (dbRes.ok) setDbData(await dbRes.json());
      }
      setLoading(false);
    }
    load();
  }, [jobId]);

  if (loading) return <p className="text-zinc-500 text-sm">Loading…</p>;
  if (!status) return <p className="text-red-400 text-sm">Job not found.</p>;

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center gap-3 mb-8">
        <a href="/jobs" className="text-zinc-500 hover:text-white text-sm transition-colors">← Jobs</a>
        <span className="text-zinc-700">/</span>
        <p className="text-sm font-mono text-zinc-400 truncate">{jobId}</p>
      </div>

      {/* Status */}
      <div className="rounded-xl border border-zinc-800 p-5 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-medium text-zinc-400">Job Status</h2>
          <StatusBadge status={status.status} />
        </div>
        {status.status === "FAILED" && (
          <p className="text-sm text-red-400">{status.error}</p>
        )}
        {status.summary && (
          <div className="grid grid-cols-4 gap-3 mt-2">
            {Object.entries(status.summary as Record<string, unknown>)
              .filter(([, v]) => typeof v === "number")
              .map(([k, v]) => (
                <Stat key={k} label={k.replace(/_/g, " ")} value={v as number} />
              ))}
          </div>
        )}
      </div>

      {dbData && (
        <>
          {/* Shift + Worker */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="rounded-xl border border-zinc-800 p-5">
              <h2 className="text-sm font-medium text-zinc-400 mb-3">Shift</h2>
              <Row label="Date" value={dbData.shift.date} />
              <Row label="Status" value={dbData.shift.status} />
              <Row label="Violations" value={String(dbData.shift.violation_count)} />
              <Row label="Started" value={new Date(dbData.shift.started_at).toLocaleTimeString()} />
              {dbData.shift.ended_at && <Row label="Ended" value={new Date(dbData.shift.ended_at).toLocaleTimeString()} />}
            </div>
            <div className="rounded-xl border border-zinc-800 p-5">
              <h2 className="text-sm font-medium text-zinc-400 mb-3">Worker</h2>
              {dbData.worker ? (
                <>
                  <Row label="Name" value={dbData.worker.display_name ?? "—"} />
                  <Row label="ID" value={dbData.worker.id.slice(0, 8) + "…"} />
                  <Row label="Total violations" value={String(dbData.worker.total_violations)} />
                </>
              ) : <p className="text-zinc-600 text-sm">No worker linked</p>}
            </div>
          </div>

          {/* Events table */}
          <div className="rounded-xl border border-zinc-800 overflow-hidden">
            <div className="px-5 py-4 border-b border-zinc-800 flex items-center justify-between">
              <h2 className="text-sm font-medium text-zinc-400">Safety Events ({dbData.total_events})</h2>
            </div>
            {dbData.events.length === 0 ? (
              <p className="text-zinc-600 text-sm p-5">No events recorded.</p>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs text-zinc-500 border-b border-zinc-800">
                    <th className="text-left px-5 py-3">Timestamp</th>
                    <th className="text-left px-5 py-3">Source</th>
                    <th className="text-left px-5 py-3">Violation</th>
                    <th className="text-left px-5 py-3">Severity</th>
                    <th className="text-left px-5 py-3">Details</th>
                  </tr>
                </thead>
                <tbody>
                  {dbData.events.map((e) => (
                    <>
                      <tr
                        key={e.event_id}
                        onClick={() => setExpanded(expanded === e.event_id ? null : e.event_id)}
                        className="border-b border-zinc-800/50 hover:bg-zinc-900 cursor-pointer transition-colors"
                      >
                        <td className="px-5 py-3 font-mono text-zinc-300">{e.video_timestamp.toFixed(2)}s</td>
                        <td className="px-5 py-3 text-zinc-400">{e.camera_source}</td>
                        <td className="px-5 py-3 text-white">{e.violation_type?.replace(/_/g, " ") ?? "—"}</td>
                        <td className="px-5 py-3"><SeverityBadge severity={e.severity} /></td>
                        <td className="px-5 py-3 text-zinc-500 text-xs">click to expand</td>
                      </tr>
                      {expanded === e.event_id && e.metadata && (
                        <tr key={`${e.event_id}-meta`} className="border-b border-zinc-800/50 bg-zinc-900/50">
                          <td colSpan={5} className="px-5 py-3">
                            <pre className="text-xs text-zinc-400 overflow-auto">{JSON.stringify(e.metadata, null, 2)}</pre>
                          </td>
                        </tr>
                      )}
                    </>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
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
    <span className={`text-xs font-medium px-2.5 py-1 rounded-lg border ${colors[status] ?? ""}`}>
      {status}
    </span>
  );
}

function SeverityBadge({ severity }: { severity: string | null }) {
  const colors: Record<string, string> = {
    HIGH: "text-red-400",
    CRITICAL: "text-red-500 font-semibold",
    MEDIUM: "text-yellow-400",
    LOW: "text-zinc-400",
  };
  return <span className={`text-xs ${colors[severity ?? ""] ?? "text-zinc-500"}`}>{severity ?? "—"}</span>;
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg bg-zinc-900 p-3">
      <p className="text-xs text-zinc-500 mb-0.5">{label}</p>
      <p className="text-xl font-semibold text-white">{value}</p>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between py-1.5 border-b border-zinc-800/50 last:border-0">
      <span className="text-xs text-zinc-500">{label}</span>
      <span className="text-xs text-white">{value}</span>
    </div>
  );
}
