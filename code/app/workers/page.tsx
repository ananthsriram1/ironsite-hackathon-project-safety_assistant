export default function WorkersPage() {
  return (
    <div>
      <div style={{ marginBottom: "28px" }}>
        <div className="page-eyebrow">Registry</div>
        <h1 className="page-title">Workers</h1>
        <p className="page-sub">Worker profiles · upcoming in next release</p>
      </div>

      {/* Coming soon notice */}
      <div className="card" style={{ padding: "48px 32px", textAlign: "center", marginBottom: "20px" }}>
        <div style={{
          width: "52px",
          height: "52px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          margin: "0 auto 18px",
          background: "var(--surface-2)",
          border: "1px solid var(--border)",
          borderRadius: "50%",
        }}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="8" r="4" stroke="var(--text-muted)" strokeWidth="1.5" />
            <path d="M4 20c0-4.418 3.582-6 8-6s8 1.582 8 6" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </div>
        <div style={{ fontSize: "15px", fontWeight: 700, color: "var(--text)", marginBottom: "8px" }}>Worker Profiles Coming Soon</div>
        <div style={{ fontSize: "12px", color: "var(--text-muted)", maxWidth: "380px", margin: "0 auto", lineHeight: 1.6 }}>
          Individual dashboards with violation history, PPE compliance rates, MSD risk scores, and shift summaries per worker.
        </div>
      </div>

      {/* Wireframe preview */}
      <div style={{ opacity: 0.35, pointerEvents: "none" }}>
        <div className="section-label">Preview</div>
        <div className="row-list">
          {[
            { name: "Worker A-001", age: "2h ago", viol: 3, comp: 94 },
            { name: "Worker B-042", age: "1d ago", viol: 7, comp: 81 },
            { name: "Worker C-017", age: "3d ago", viol: 1, comp: 97 },
          ].map(({ name, age, viol, comp }) => (
            <div key={name} className="row-item" style={{ justifyContent: "space-between" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                <div style={{ width: "30px", height: "30px", borderRadius: "50%", background: "var(--surface-2)", border: "1px solid var(--border)", flexShrink: 0 }} />
                <div>
                  <div style={{ fontSize: "13px", color: "var(--text)", fontWeight: 500, marginBottom: "2px" }}>{name}</div>
                  <div style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>Last seen · {age}</div>
                </div>
              </div>
              <div style={{ display: "flex", gap: "20px", alignItems: "center" }}>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: "8px", fontFamily: "var(--font-mono)", letterSpacing: "0.1em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "2px" }}>Violations</div>
                  <div style={{ fontSize: "15px", fontFamily: "var(--font-mono)", fontWeight: 800, color: "var(--red)" }}>{viol}</div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: "8px", fontFamily: "var(--font-mono)", letterSpacing: "0.1em", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "2px" }}>Compliance</div>
                  <div style={{ fontSize: "15px", fontFamily: "var(--font-mono)", fontWeight: 800, color: "var(--emerald)" }}>{comp}%</div>
                </div>
                <span className="badge badge-completed">ACTIVE</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
