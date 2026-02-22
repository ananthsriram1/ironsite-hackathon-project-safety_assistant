const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

export const api = {
  health: () => apiFetch<{ status: string }>("/health"),

  workers: {
    list: (siteId: string) => apiFetch<Worker[]>(`/workers?site_id=${siteId}`),
    get:  (workerId: string) => apiFetch<Worker>(`/workers/${workerId}`),
  },

  shifts: {
    get: (shiftId: string) => apiFetch<Shift>(`/shifts/${shiftId}`),
  },

  events: {
    list: (params: { workerId?: string; shiftId?: string; eventCategory?: string }) => {
      const q = new URLSearchParams();
      if (params.workerId)      q.set("worker_id", params.workerId);
      if (params.shiftId)       q.set("shift_id", params.shiftId);
      if (params.eventCategory) q.set("event_category", params.eventCategory);
      return apiFetch<SafetyEvent[]>(`/events?${q}`);
    },
  },

  jobs: {
    get: (jobId: string) => apiFetch<Job>(`/jobs/${jobId}`),
  },
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Worker {
  worker_id: string;
  display_name: string | null;
  status: "ACTIVE" | "INACTIVE";
  total_violations: number;
  total_compliant: number;
  avg_msd_risk: number;
  last_seen_at: string | null;
}

export interface Shift {
  shift_id: string;
  worker_id: string;
  date: string;
  status: "IN_PROGRESS" | "COMPLETED";
  msd_risk_score: number | null;
  violation_count: number;
  compliant_count: number;
  events: SafetyEvent[];
}

export interface SafetyEvent {
  event_id: string;
  shift_id: string;
  worker_id: string;
  camera_source: "POV" | "WALL_CAM";
  event_category: "VIOLATION" | "COMPLIANT";
  violation_type: string | null;
  severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | null;
  video_timestamp: number;
  clip_path: string | null;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

export interface Job {
  job_id: string;
  status: "PROCESSING" | "COMPLETED" | "FAILED";
  events_written: number;
  error: string | null;
  camera_source: string | null;
  shift_id?: string;
  summary?: {
    frames_processed: number;
    frames_with_hazards: number;
    workers_tracked: number;
    hazard_counts: Record<string, number>;
    detected_classes: string[];
    worker_compliance: Record<string, { ppe_confirmed: string[]; ppe_missing: string[]; compliant: boolean }>;
  };
}
