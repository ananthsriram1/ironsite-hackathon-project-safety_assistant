"use client";

import { useEffect, useRef, useState } from "react";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type JobStatus = {
  job_id: string;
  status: "PROCESSING" | "COMPLETED" | "FAILED";
  events_written: number;
  error: string | null;
  summary?: {
    frames_processed: number;
    frames_with_hazards: number;
    hazard_counts: Record<string, number>;
    detected_classes: string[];
    video_fps: number;
  };
};

export default function IngestPage() {
  const [mode, setMode] = useState<"wall-cam" | "pov">("wall-cam");
  const [siteId, setSiteId] = useState("");
  const [workerId, setWorkerId] = useState("");
  const [date, setDate] = useState(new Date().toISOString().slice(0, 10));
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<JobStatus[]>([]);
  const pollRef = useRef<Record<string, NodeJS.Timeout>>({});

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!file || !siteId || !date) return;
    setSubmitting(true);
    setError(null);

    try {
      const form = new FormData();
      form.append("video", file);
      form.append("site_id", siteId);
      form.append("date", date);
      form.append("worker_id", workerId);

      const res = await fetch(`${BASE}/process/${mode}`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const { job_id } = await res.json();

      const newJob: JobStatus = { job_id, status: "PROCESSING", events_written: 0, error: null };
      setJobs((prev) => [newJob, ...prev]);
      startPolling(job_id);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setSubmitting(false);
    }
  }

  function startPolling(job_id: string) {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${BASE}/jobs/${job_id}`);
        if (!res.ok) return;
        const data: JobStatus = await res.json();
        setJobs((prev) => prev.map((j) => (j.job_id === job_id ? data : j)));
        if (data.status !== "PROCESSING") {
          clearInterval(interval);
          delete pollRef.current[job_id];
        }
      } catch {}
    }, 2000);
    pollRef.current[job_id] = interval;
  }

  useEffect(() => {
    return () => Object.values(pollRef.current).forEach(clearInterval);
  }, []);

  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-2xl font-semibold text-white mb-1">Ingest</h1>
      <p className="text-sm text-zinc-500 mb-8">Upload footage for processing</p>

      <form onSubmit={submit} className="rounded-xl border border-zinc-800 p-6 space-y-5 mb-10">
        {/* Mode toggle */}
        <div className="flex gap-2">
          {(["wall-cam", "pov"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                mode === m
                  ? "bg-zinc-100 text-zinc-900"
                  : "bg-zinc-800 text-zinc-400 hover:text-white"
              }`}
            >
              {m === "wall-cam" ? "Wall Cam" : "POV"}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Field label="Site ID" value={siteId} onChange={setSiteId} placeholder="site-uuid" />
          <Field label="Date" value={date} onChange={setDate} type="date" />
        </div>

        <Field label="Worker ID" value={workerId} onChange={setWorkerId} placeholder="worker-uuid" />

        {/* File picker */}
        <div>
          <label className="block text-xs text-zinc-400 mb-1.5">Video file</label>
          <input
            type="file"
            accept="video/*,.mp4,.mov,.avi"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="block w-full text-sm text-zinc-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-zinc-800 file:text-zinc-200 hover:file:bg-zinc-700 cursor-pointer"
          />
          {file && <p className="text-xs text-zinc-500 mt-1">{file.name} — {(file.size / 1e6).toFixed(1)} MB</p>}
        </div>

        {error && <p className="text-sm text-red-400">{error}</p>}

        <button
          type="submit"
          disabled={submitting || !file || !siteId}
          className="w-full py-2 rounded-lg bg-white text-zinc-900 text-sm font-semibold disabled:opacity-40 hover:bg-zinc-100 transition-colors"
        >
          {submitting ? "Uploading…" : "Start processing"}
        </button>
      </form>

      {/* Job list */}
      {jobs.length > 0 && (
        <section>
          <h2 className="text-sm font-medium text-zinc-400 mb-3">Recent jobs</h2>
          <div className="space-y-3">
            {jobs.map((job) => (
              <JobCard key={job.job_id} job={job} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function Field({
  label, value, onChange, placeholder = "", type = "text",
}: {
  label: string; value: string; onChange: (v: string) => void; placeholder?: string; type?: string;
}) {
  return (
    <div>
      <label className="block text-xs text-zinc-400 mb-1.5">{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-zinc-500"
      />
    </div>
  );
}

function JobCard({ job }: { job: JobStatus }) {
  const statusColor =
    job.status === "COMPLETED" ? "text-emerald-400" :
    job.status === "FAILED" ? "text-red-400" :
    "text-yellow-400";

  return (
    <div className="rounded-xl border border-zinc-800 p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-zinc-500 font-mono">{job.job_id.slice(0, 8)}…</span>
        <span className={`text-xs font-medium ${statusColor}`}>
          {job.status === "PROCESSING" && <span className="animate-pulse">● </span>}
          {job.status}
        </span>
      </div>

      {job.status === "PROCESSING" && (
        <p className="text-xs text-zinc-500">Running detection pipeline…</p>
      )}

      {job.status === "FAILED" && (
        <p className="text-xs text-red-400">{job.error}</p>
      )}

      {job.status === "COMPLETED" && job.summary && (
        <div className="grid grid-cols-3 gap-3 mt-2">
          <Stat label="Frames processed" value={job.summary.frames_processed} />
          <Stat label="Hazard frames" value={job.summary.frames_with_hazards} />
          <Stat label="Detected classes" value={job.summary.detected_classes.length} />
          {Object.entries(job.summary.hazard_counts).map(([k, v]) => (
            <Stat key={k} label={k.replace(/_/g, " ")} value={v} />
          ))}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg bg-zinc-900 p-2.5">
      <p className="text-xs text-zinc-500 mb-0.5">{label}</p>
      <p className="text-lg font-semibold text-white">{value}</p>
    </div>
  );
}
