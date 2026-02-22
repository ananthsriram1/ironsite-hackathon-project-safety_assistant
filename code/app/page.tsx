"use client";

import { useEffect, useState } from "react";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type JobEntry = {
  job_id: string;
  status: string;
  events_written: number;
  camera_source: string | null;
  shift?: {
    worker: string | null;
    site: string | null;
    date: string;
    violation_count: number;
    event_count: number;
    status: string;
  };
};

export default function DashboardPage() {
  const [connected, setConnected] = useState<boolean | null>(null);
  const [jobs, setJobs] = useState<JobEntry[]>([]);

  useEffect(() => {
    fetch(`${BASE}/health`)
      .then((r) => { if (r.ok) setConnected(true); else setConnected(false); })
      .catch(() => setConnected(false));

    fetch(`${BASE}/jobs`)
      .then((r) => r.json())
      .then(setJobs)
      .catch(() => {});
  }, []);

  const completed = jobs.filter((j) => j.status === "COMPLETED");
  const totalViolations = completed.reduce((s, j) => s + (j.shift?.violation_count ?? 0), 0);
  const totalEvents    = completed.reduce((s, j) => s + (j.shift?.event_count ?? 0), 0);
  const recentJobs = completed.slice(0, 5);

  const processing = jobs.filter(j => j.status === "PROCESSING").length;
  const failed = jobs.filter(j => j.status === "FAILED").length;

  return (
    <div>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: "28px" }}>
        <div>
          <div className="page-eyebrow">Operations Center</div>
          <h1 className="page-title">Dashboard</h1>
          <p className="page-sub">Safety overview · all processed jobs</p>
        </div>
        <BackendStatus connected={connected} />
      </div>

      {/* KPI Row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px", marginBottom: "24px" }}>
        <div className="stat-card" style={{ borderLeftColor: "var(--accent)" }}>
          <div className="stat-label">Jobs Processed</div>
          <div className="stat-value">{completed.length}</div>
        </div>
        <div className="stat-card" style={{ borderLeftColor: "var(--red)" }}>
          <div className="stat-label">Violations Detected</div>
          <div className="stat-value" style={{ color: totalViolations > 0 ? "var(--red)" : "var(--text)" }}>
            {totalViolations}
          </div>
        </div>
        <div className="stat-card" style={{ borderLeftColor: "var(--blue)" }}>
          <div className="stat-label">Events in DB</div>
          <div className="stat-value">{totalEvents}</div>
        </div>
      </div>

      {/* Bottom panels */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
        {/* Recent Jobs */}
        <div className="card">
          <div className="card-head">
            <span className="card-label">Recent Jobs</span>
            <a
              href="/jobs"
              style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--accent)", textDecoration: "none", letterSpacing: "0.06em" }}
            >
              ALL JOBS →
            </a>
          </div>
          {recentJobs.length === 0 ? (
            <div className="card-body" style={{ fontSize: "12px", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>
              No completed jobs yet.
            </div>
          ) : (
            <div>
              {recentJobs.map((j) => (
                <a
                  key={j.job_id}
                  href={`/jobs/${j.job_id}`}
                  style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 16px", borderBottom: "1px solid var(--border)", textDecoration: "none", transition: "background 0.1s" }}
                  onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-2)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                >
                  <div>
                    <div style={{ fontSize: "13px", color: "var(--text)", marginBottom: "2px", fontWeight: 500 }}>
                      {j.shift?.worker ?? "Unknown"} — {j.shift?.site ?? j.job_id.slice(0, 8)}
                    </div>
                    <div style={{ fontSize: "10px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                      {j.shift?.date} · {j.camera_source ?? "—"}
                    </div>
                  </div>
                  <div style={{ fontSize: "11px", fontFamily: "var(--font-mono)", fontWeight: 700, color: (j.shift?.violation_count ?? 0) > 0 ? "var(--red)" : "var(--text-muted)" }}>
                    {j.shift?.violation_count ?? 0} viol.
                  </div>
                </a>
              ))}
            </div>
          )}
        </div>

        {/* Processing Status */}
        <div className="card">
          <div className="card-head">
            <span className="card-label">Processing Status</span>
            <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>{jobs.length} total</span>
          </div>
          <div>
            {[
              { label: "Completed", count: completed.length, color: "var(--emerald)", bg: "var(--emerald-bg)", border: "var(--emerald-border)" },
              { label: "Processing", count: processing,      color: "var(--amber)",   bg: "var(--amber-bg)",   border: "var(--amber-border)" },
              { label: "Failed",     count: failed,          color: "var(--red)",     bg: "var(--red-bg)",     border: "var(--red-border)"   },
            ].map(({ label, count, color, bg, border }) => (
              <div
                key={label}
                style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 16px", borderBottom: "1px solid var(--border)" }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  <div style={{ width: "8px", height: "8px", borderRadius: "2px", background: color, flexShrink: 0 }} />
                  <span style={{ fontSize: "13px", color: "var(--text-muted)", fontWeight: 500 }}>{label}</span>
                </div>
                <span style={{
                  fontSize: "20px",
                  fontWeight: 800,
                  fontFamily: "var(--font-mono)",
                  color,
                  background: bg,
                  border: `1px solid ${border}`,
                  padding: "0 12px",
                  borderRadius: "5px",
                  lineHeight: "32px",
                }}>
                  {count}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function BackendStatus({ connected }: { connected: boolean | null }) {
  if (connected === null) {
    return (
      <div className="status-pill" style={{ color: "var(--text-muted)" }}>
        <span className="status-dot animate-pulse" style={{ background: "var(--text-dim)" }} />
        Connecting
      </div>
    );
  }
  if (connected) {
    return (
      <div className="status-pill" style={{ color: "var(--emerald)" }}>
        <span className="status-dot" style={{ background: "var(--emerald)", boxShadow: "0 0 6px var(--emerald)" }} />
        Backend Online
      </div>
    );
  }
  return (
    <div className="status-pill" style={{ color: "var(--red)" }}>
      <span className="status-dot" style={{ background: "var(--red)" }} />
      Backend Offline
    </div>
  );
}
