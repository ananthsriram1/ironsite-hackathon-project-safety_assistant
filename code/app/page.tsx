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

  // Top violation types across all jobs (from summary hazard_counts would need more data — use event counts for now)
  const recentJobs = completed.slice(0, 5);

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-white">Site Dashboard</h1>
          <p className="text-sm text-zinc-500 mt-1">Safety overview across all processed jobs</p>
        </div>
        <BackendStatus connected={connected} />
      </div>

      <div className="grid grid-cols-3 gap-4 mb-8">
        <StatCard label="Jobs Processed" value={String(completed.length)} />
        <StatCard label="Violations Detected" value={String(totalViolations)} />
        <StatCard label="Events in DB" value={String(totalEvents)} />
      </div>

      <div className="grid grid-cols-2 gap-6">
        <section className="rounded-xl border border-zinc-800 p-5">
          <h2 className="text-sm font-medium text-zinc-400 mb-4">Recent Jobs</h2>
          {recentJobs.length === 0 ? (
            <p className="text-zinc-600 text-sm">No completed jobs yet.</p>
          ) : (
            <div className="space-y-2">
              {recentJobs.map((j) => (
                <a
                  key={j.job_id}
                  href={`/jobs/${j.job_id}`}
                  className="flex items-center justify-between py-2 border-b border-zinc-800/50 last:border-0 hover:opacity-75 transition-opacity"
                >
                  <div>
                    <p className="text-sm text-white">{j.shift?.worker ?? "Unknown"} — {j.shift?.site ?? j.job_id.slice(0, 8)}</p>
                    <p className="text-xs text-zinc-500">{j.shift?.date} · {j.camera_source}</p>
                  </div>
                  <span className="text-xs text-red-400 font-medium">{j.shift?.violation_count ?? 0} violations</span>
                </a>
              ))}
            </div>
          )}
        </section>

        <section className="rounded-xl border border-zinc-800 p-5">
          <h2 className="text-sm font-medium text-zinc-400 mb-4">Processing Status</h2>
          <div className="space-y-2">
            {[
              { label: "Completed", count: jobs.filter(j => j.status === "COMPLETED").length, color: "text-emerald-400" },
              { label: "Processing", count: jobs.filter(j => j.status === "PROCESSING").length, color: "text-yellow-400" },
              { label: "Failed", count: jobs.filter(j => j.status === "FAILED").length, color: "text-red-400" },
            ].map(({ label, count, color }) => (
              <div key={label} className="flex items-center justify-between py-2 border-b border-zinc-800/50 last:border-0">
                <span className="text-sm text-zinc-400">{label}</span>
                <span className={`text-sm font-semibold ${color}`}>{count}</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

function BackendStatus({ connected }: { connected: boolean | null }) {
  if (connected === null)
    return <span className="flex items-center gap-2 text-xs text-zinc-500"><span className="h-2 w-2 rounded-full bg-zinc-600 animate-pulse" />Connecting...</span>;
  if (connected)
    return <span className="flex items-center gap-2 text-xs text-emerald-400"><span className="h-2 w-2 rounded-full bg-emerald-400" />Backend connected</span>;
  return <span className="flex items-center gap-2 text-xs text-red-400"><span className="h-2 w-2 rounded-full bg-red-400" />Backend offline</span>;
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-zinc-800 p-5">
      <p className="text-xs text-zinc-500 mb-1">{label}</p>
      <p className="text-3xl font-semibold text-white">{value}</p>
    </div>
  );
}
