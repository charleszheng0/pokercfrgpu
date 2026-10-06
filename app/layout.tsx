import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = {
  title: "Poker GTO Solver",
  description: "Kuhn Poker CFR solver lab",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
