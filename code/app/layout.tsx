import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "IronSite",
  description: "Construction site safety intelligence",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        <div className="app-shell">
          {/* Sidebar */}
          <aside className="sidebar">
            {/* Brand */}
            <div className="sidebar-brand">
              <div className="brand-icon">
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                  <path d="M9 1.5L16 5.25V12.75L9 16.5L2 12.75V5.25L9 1.5Z" stroke="#f97316" strokeWidth="1.4" fill="none" />
                  <path d="M9 5.5L12.5 7.5V11.5L9 13.5L5.5 11.5V7.5L9 5.5Z" fill="#f97316" fillOpacity="0.25" />
                  <circle cx="9" cy="9" r="1.5" fill="#f97316" />
                </svg>
              </div>
              <div>
                <div className="brand-name">IronSite</div>
                <div className="brand-tag">Safety Intelligence</div>
              </div>
            </div>

            {/* Nav */}
            <nav className="sidebar-nav">
              <a href="/" className="nav-item">
                <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                  <rect x="1" y="1" width="5.5" height="5.5" rx="1" stroke="currentColor" strokeWidth="1.2" />
                  <rect x="8.5" y="1" width="5.5" height="5.5" rx="1" stroke="currentColor" strokeWidth="1.2" />
                  <rect x="1" y="8.5" width="5.5" height="5.5" rx="1" stroke="currentColor" strokeWidth="1.2" />
                  <rect x="8.5" y="8.5" width="5.5" height="5.5" rx="1" stroke="currentColor" strokeWidth="1.2" />
                </svg>
                Dashboard
              </a>

              <a href="/jobs" className="nav-item">
                <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                  <rect x="1" y="2.5" width="13" height="10" rx="1" stroke="currentColor" strokeWidth="1.2" />
                  <path d="M3.5 6.5h8M3.5 9h5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                </svg>
                Jobs
              </a>

              <a href="/ingest" className="nav-item">
                <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                  <path d="M7.5 1.5v8M4.5 7l3 3 3-3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M2 11.5h11" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                </svg>
                Ingest
              </a>

              <a href="/workers" className="nav-item">
                <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                  <circle cx="7.5" cy="5" r="3" stroke="currentColor" strokeWidth="1.2" />
                  <path d="M1.5 13.5c0-3.314 2.686-5 6-5s6 1.686 6 5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                </svg>
                Workers
              </a>
            </nav>

            {/* Footer */}
            <div className="sidebar-footer">
              <div className="footer-sys">System</div>
              <div className="footer-ver">YOLO11 · Qwen3-VL · v0.1</div>
            </div>
          </aside>

          {/* Main */}
          <main className="main-content">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
