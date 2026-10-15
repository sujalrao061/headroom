import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Headroom — Make it fit. Prove it works.",
  description: "Evidence-backed GPU memory repairs for PyTorch workloads.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
