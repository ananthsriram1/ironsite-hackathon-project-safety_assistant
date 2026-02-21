"use client";

import { useEffect, useState } from "react";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Job = {
  job_id: string;
  status: "PROCESSING" | "COMPLETED" | "FAILED";
  events_written: number;
  error: string | null;
  shift?: {
    shift_id: string;
    date: string;
    worker: string | null;
    violation_count: number;
    status: string;
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
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-white">Jobs</h1>
          <p className="text-sm text-zinc-500 mt-1">Processing history — refreshes every 3s</p>
        </div>
        <button onClick={load} className="text-xs text-zinc-400 hover:text-white border border-zinc-700 rounded-lg px-3 py-1.5 transition-colors">
          Refresh
        </button>
      </div>

      {loading && <p className="text-zinc-500 text-sm">Loading…</p>}
      {!loading && jobs.length === 0 && (
        <p className="text-zinc-600 text-sm">No jobs yet — upload a video from the Ingest page.</p>
      )}

      <div className="space-y-3">
        {jobs.map((job) => (
          <a key={job.job_id} href={`/jobs/${job.job_id}`} className="block rounded-xl border border-zinc-800 p-5 hover:border-zinc-600 transition-colors">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-xs font-mono text-zinc-400 mb-1">{job.job_id}</p>
                {job.shift && (
                  <p className="text-sm text-white">
                    {job.shift.worker ?? "Unknown worker"} — {job.shift.date}
                  </p>
                )}
              </div>
              <StatusBadge status={job.status} />
            </div>

            {job.status === "COMPLETED" && (
              <div className="flex gap-6 mt-3 pt-3 border-t border-zinc-800">
                <Stat label="Events written" value={job.events_written} />
                {job.shift && <Stat label="Violations in DB" value={job.shift.violation_count} />}
              </div>
            )}
            {job.status === "FAILED" && (
              <p className="text-xs text-red-400 mt-2 truncate">{job.error}</p>
            )}
          </a>
        ))}
      </div>
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
    <span className={`shrink-0 text-xs font-medium px-2.5 py-1 rounded-lg border ${colors[status] ?? ""}`}>
      {status === "PROCESSING" && <span className="animate-pulse">● </span>}
      {status}
    </span>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <p className="text-xs text-zinc-500">{label}</p>
      <p className="text-lg font-semibold text-white">{value}</p>
    </div>
  );
}
