import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MEDORBY — Privacy-First Medical AI Council",
  description:
    "MEDORBY uses a 3-stage LLM Council powered by Groq and GPT-OSS 120B to deliver consensus-driven medical reasoning — with zero PII leaving your device.",
  keywords: ["medical AI", "privacy", "LLM council", "federated learning", "Groq", "differential privacy"],
  openGraph: {
    title: "MEDORBY — Privacy-First Medical AI",
    description: "Consensus-driven medical reasoning. PII never leaves your device.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="noise">{children}</body>
    </html>
  );
}
