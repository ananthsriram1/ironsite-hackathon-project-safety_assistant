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
    <div style={{ maxWidth: "700px" }}>
      {/* Header */}
      <div style={{ marginBottom: "28px" }}>
        <div className="page-eyebrow">Video Analysis</div>
        <h1 className="page-title">Ingest</h1>
        <p className="page-sub">Upload footage for AI safety analysis pipeline</p>
      </div>

      {/* Form */}
      <form onSubmit={submit} className="card" style={{ marginBottom: "24px" }}>
        <div className="card-head">
          <span className="card-label">New Analysis Job</span>
          {/* Mode toggle */}
          <div style={{ display: "flex", gap: "4px" }}>
            {(["wall-cam", "pov"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={`mode-btn${mode === m ? " active" : ""}`}
              >
                {m === "wall-cam" ? "WALL-CAM" : "POV"}
              </button>
            ))}
          </div>
        </div>

        <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
            <div>
              <label className="field-label">Site ID</label>
              <input
                className="field-input"
                type="text"
                value={siteId}
                onChange={(e) => setSiteId(e.target.value)}
                placeholder="site-uuid"
              />
            </div>
            <div>
              <label className="field-label">Date</label>
              <input
                className="field-input"
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
              />
            </div>
          </div>

          <div>
            <label className="field-label">Worker ID</label>
            <input
              className="field-input"
              type="text"
              value={workerId}
              onChange={(e) => setWorkerId(e.target.value)}
              placeholder="worker-uuid (optional)"
            />
          </div>

          {/* File drop zone */}
          <div>
            <label className="field-label">Video File</label>
            <div className="drop-zone">
              <input
                type="file"
                accept="video/*,.mp4,.mov,.avi"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              <div style={{ pointerEvents: "none" }}>
                {file ? (
                  <div>
                    <div style={{ fontSize: "28px", marginBottom: "8px" }}>🎬</div>
                    <div style={{ fontSize: "13px", color: "var(--accent)", fontWeight: 700 }}>{file.name}</div>
                    <div style={{ fontSize: "10px", color: "var(--text-muted)", fontFamily: "var(--font-mono)", marginTop: "4px" }}>
                      {(file.size / 1e6).toFixed(1)} MB · {mode === "wall-cam" ? "Wall Camera" : "POV Camera"}
                    </div>
                  </div>
                ) : (
                  <div>
                    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" style={{ margin: "0 auto 10px", display: "block", opacity: 0.4 }}>
                      <path d="M14 4v14M8 12l6 6 6-6" stroke="var(--text)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                      <path d="M4 22h20" stroke="var(--text)" strokeWidth="1.5" strokeLinecap="round" />
                    </svg>
                    <div style={{ fontSize: "13px", color: "var(--text-muted)" }}>Drop video or click to browse</div>
                    <div style={{ fontSize: "10px", color: "var(--text-dim)", fontFamily: "var(--font-mono)", marginTop: "4px" }}>
                      MP4 · MOV · AVI
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {error && (
            <div style={{
              fontSize: "12px",
              color: "var(--red)",
              fontFamily: "var(--font-mono)",
              padding: "10px 14px",
              background: "var(--red-bg)",
              border: "1px solid var(--red-border)",
              borderRadius: "6px",
            }}>
              ERROR: {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting || !file || !siteId}
            className="btn-primary"
          >
            {submitting ? (
              <>
                <span className="spinner" style={{ borderTopColor: "#fff" }} />
                Uploading…
              </>
            ) : "Run Safety Analysis"}
          </button>
        </div>
      </form>

      {/* Session jobs */}
      {jobs.length > 0 && (
        <div>
          <div className="section-label">Session Jobs</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            {jobs.map((job) => (
              <JobCard key={job.job_id} job={job} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function JobCard({ job }: { job: JobStatus }) {
  const content = (
    <div className="card" style={{ cursor: job.status === "COMPLETED" ? "pointer" : "default" }}>
      <div style={{ padding: "12px 16px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
          {job.job_id.slice(0, 14)}…
        </span>
        {job.status === "PROCESSING" ? (
          <div style={{ display: "flex", alignItems: "center", gap: "7px", fontSize: "9px", fontFamily: "var(--font-mono)", color: "var(--amber)", fontWeight: 700, letterSpacing: "0.1em" }}>
            <span className="spinner" />
            PROCESSING
          </div>
        ) : (
          <span className={`badge badge-${job.status.toLowerCase()}`}>{job.status}</span>
        )}
      </div>

      {job.status === "PROCESSING" && (
        <div style={{ padding: "0 16px 12px", fontSize: "11px", color: "var(--text-muted)" }}>
          Running YOLO + VLM detection pipeline…
        </div>
      )}

      {job.status === "FAILED" && (
        <div style={{ padding: "0 16px 12px", fontSize: "11px", color: "var(--red)", fontFamily: "var(--font-mono)" }}>
          {job.error}
        </div>
      )}

      {job.status === "COMPLETED" && job.summary && (
        <div style={{ borderTop: "1px solid var(--border)", padding: "12px 16px", display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "8px" }}>
          <div className="mini-stat">
            <div className="mini-stat-label">Frames</div>
            <div className="mini-stat-value">{job.summary.frames_processed}</div>
          </div>
          <div className="mini-stat">
            <div className="mini-stat-label">Hazard Frames</div>
            <div className="mini-stat-value" style={{ color: "var(--amber)" }}>{job.summary.frames_with_hazards}</div>
          </div>
          <div className="mini-stat">
            <div className="mini-stat-label">Classes</div>
            <div className="mini-stat-value">{job.summary.detected_classes.length}</div>
          </div>
          {Object.entries(job.summary.hazard_counts).map(([k, v]) => (
            <div key={k} className="mini-stat">
              <div className="mini-stat-label">{k.replace(/_/g, " ")}</div>
              <div className="mini-stat-value" style={{ color: "var(--red)" }}>{v}</div>
            </div>
          ))}
        </div>
      )}

      {job.status === "COMPLETED" && (
        <div style={{ borderTop: "1px solid var(--border)", padding: "8px 16px", fontSize: "9px", color: "var(--accent)", fontFamily: "var(--font-mono)", letterSpacing: "0.1em" }}>
          VIEW FULL REPORT →
        </div>
      )}
    </div>
  );

  if (job.status === "COMPLETED") {
    return <a href={`/jobs/${job.job_id}`} style={{ textDecoration: "none" }}>{content}</a>;
  }
  return content;
}
