"use client";

import { useEffect, useState } from "react";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Job = {
  job_id: string;
  status: "PROCESSING" | "COMPLETED" | "FAILED";
  events_written: number;
  error: string | null;
  camera_source: string | null;
  shift?: {
    shift_id: string;
    date: string;
    worker_id: string;
    worker: string | null;
    site: string | null;
    violation_count: number;
    event_count: number;
    status: string;
    started_at: string;
    ended_at: string | null;
  };
};

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);

  async function load() {
    try {
      const res = await fetch(`${BASE}/jobs`);
      if (res.ok) setJobs(await res.json());
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, []);

  return (
    <div>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: "28px" }}>
        <div>
          <div className="page-eyebrow">Processing History</div>
          <h1 className="page-title">Jobs</h1>
          <p className="page-sub">Auto-refreshes every 3s</p>
        </div>
        <button onClick={load} className="btn-ghost">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M10.5 6A4.5 4.5 0 112 6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            <path d="M10.5 3v3h-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Refresh
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--text-muted)", fontSize: "12px" }}>
          <span className="spinner" /> Loading…
        </div>
      )}

      {/* Empty */}
      {!loading && jobs.length === 0 && (
        <div className="card" style={{ padding: "40px 24px", textAlign: "center" }}>
          <div style={{ fontSize: "12px", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>
            No jobs yet — upload a video from Ingest
          </div>
        </div>
      )}

      {/* Table */}
      {jobs.length > 0 && (
        <div className="card">
          {/* Head */}
          <div
            className="data-table-head"
            style={{ gridTemplateColumns: "1fr 130px 70px 80px 90px 110px" }}
          >
            {["Job / Shift", "Camera", "Frames", "Events", "Violations", "Status"].map((h) => (
              <span key={h} className="th">{h}</span>
            ))}
          </div>

          {/* Rows */}
          {jobs.map((job) => (
            <a
              key={job.job_id}
              href={`/jobs/${job.job_id}`}
              className="data-row"
              style={{ gridTemplateColumns: "1fr 130px 70px 80px 90px 110px" }}
            >
              {/* Job / shift */}
              <div>
                <div style={{ fontSize: "13px", color: "var(--text)", fontWeight: 500, marginBottom: "2px" }}>
                  {job.shift
                    ? <>{job.shift.worker ?? job.shift.worker_id} — {job.shift.site ?? "Unknown site"}</>
                    : <span style={{ color: "var(--text-muted)" }}>No shift data</span>
                  }
                </div>
                <div style={{ fontSize: "10px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                  {job.job_id.slice(0, 18)}… {job.shift ? `· ${job.shift.date}` : ""}
                </div>
              </div>

              {/* Camera */}
              <div>
                {job.camera_source && (
                  <span style={{
                    fontSize: "9px",
                    fontFamily: "var(--font-mono)",
                    fontWeight: 700,
                    letterSpacing: "0.08em",
                    color: "var(--text-muted)",
                    background: "var(--surface-2)",
                    border: "1px solid var(--border)",
                    borderRadius: "4px",
                    padding: "3px 7px",
                  }}>
                    {job.camera_source}
                  </span>
                )}
              </div>

              {/* Frames */}
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "13px", color: "var(--text-muted)" }}>
                {job.events_written}
              </div>

              {/* Events */}
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "13px", color: "var(--text)" }}>
                {job.shift?.event_count ?? "—"}
              </div>

              {/* Violations */}
              <div style={{
                fontFamily: "var(--font-mono)",
                fontSize: "13px",
                fontWeight: 700,
                color: (job.shift?.violation_count ?? 0) > 0 ? "var(--red)" : "var(--text-muted)",
              }}>
                {job.shift?.violation_count ?? "—"}
              </div>

              {/* Status */}
              <div>
                {job.status === "PROCESSING" ? (
                  <span className="badge badge-processing">
                    <span className="animate-pulse" style={{ display: "inline-block", width: "5px", height: "5px", borderRadius: "50%", background: "var(--amber)" }} />
                    PROCESSING
                  </span>
                ) : (
                  <span className={`badge badge-${job.status.toLowerCase()}`}>{job.status}</span>
                )}
              </div>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
