"use client";

import { api } from "@/lib/api";
import { useEffect, useState } from "react";

export default function DashboardPage() {
  const [connected, setConnected] = useState<boolean | null>(null);

  useEffect(() => {
    api.health()
      .then(() => setConnected(true))
      .catch(() => setConnected(false));
  }, []);

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-semibold text-white">Site Dashboard</h1>
          <p className="text-sm text-zinc-500 mt-1">End-of-day safety summary</p>
        </div>
        <BackendStatus connected={connected} />
      </div>

      <div className="grid grid-cols-3 gap-4 mb-8">
        <StatCard label="Workers on Site" value="—" />
        <StatCard label="Violations Today" value="—" />
        <StatCard label="Compliant Events" value="—" />
      </div>

      <div className="grid grid-cols-2 gap-6">
        <section className="rounded-xl border border-zinc-800 p-5">
          <h2 className="text-sm font-medium text-zinc-400 mb-4">Worker Summary</h2>
          <p className="text-zinc-600 text-sm">No shift data yet.</p>
        </section>
        <section className="rounded-xl border border-zinc-800 p-5">
          <h2 className="text-sm font-medium text-zinc-400 mb-4">Top Violations</h2>
          <p className="text-zinc-600 text-sm">No events yet.</p>
        </section>
      </div>
    </div>
  );
}

function BackendStatus({ connected }: { connected: boolean | null }) {
  if (connected === null)
    return (
      <span className="flex items-center gap-2 text-xs text-zinc-500">
        <span className="h-2 w-2 rounded-full bg-zinc-600 animate-pulse" />
        Connecting...
      </span>
    );
  if (connected)
    return (
      <span className="flex items-center gap-2 text-xs text-emerald-400">
        <span className="h-2 w-2 rounded-full bg-emerald-400" />
        Backend connected
      </span>
    );
  return (
    <span className="flex items-center gap-2 text-xs text-red-400">
      <span className="h-2 w-2 rounded-full bg-red-400" />
      Backend offline — run docker compose up in /backend
    </span>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-zinc-800 p-5">
      <p className="text-xs text-zinc-500 mb-1">{label}</p>
      <p className="text-3xl font-semibold text-white">{value}</p>
    </div>
  );
}
