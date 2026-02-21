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
  description: "Construction site safety intelligence dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-zinc-950 text-zinc-100`}>
        <nav className="border-b border-zinc-800 px-6 py-4 flex items-center gap-8">
          <span className="font-semibold tracking-tight text-white">IronSite</span>
          <a href="/" className="text-sm text-zinc-400 hover:text-white transition-colors">Dashboard</a>
          <a href="/workers" className="text-sm text-zinc-400 hover:text-white transition-colors">Workers</a>
          <a href="/ingest" className="text-sm text-zinc-400 hover:text-white transition-colors">Ingest</a>
          <a href="/jobs" className="text-sm text-zinc-400 hover:text-white transition-colors">Jobs</a>
        </nav>
        <main className="px-6 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
