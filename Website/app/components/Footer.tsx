import Link from "next/link";

const Logo = () => (
  <Link href="/" style={{ textDecoration: "none", display: "inline-flex", alignItems: "baseline", color: "#ffffff", lineHeight: 1 }}>
    <span style={{ fontFamily: "'Inter', sans-serif", fontWeight: "800", fontSize: "1.85rem", letterSpacing: "-0.06em" }}>veri</span>
    <span style={{ fontFamily: "var(--font-fraunces), serif", fontStyle: "italic", fontWeight: "400", fontSize: "2.6rem", margin: "0 0.01em 0 -0.04em", transform: "translateY(3px)", display: "inline-block" }}>t</span>
    <span style={{ position: "relative", display: "inline-block", fontFamily: "'Inter', sans-serif", fontWeight: "800", fontSize: "1.85rem", letterSpacing: "-0.06em" }}>
      a
      <span style={{ position: "absolute", top: "50%", left: "-10%", width: "120%", height: "3px", background: "#000", transform: "rotate(-25deg)" }} />
    </span>
    <span style={{ fontFamily: "'Inter', sans-serif", fontWeight: "800", fontSize: "1.85rem", letterSpacing: "-0.06em" }}>s</span>
    <span style={{ fontFamily: "'Inter', sans-serif", fontWeight: "800", fontSize: "1.85rem", color: "#38bdf8", letterSpacing: "-0.03em", marginLeft: "2px" }}>.ai</span>
  </Link>
);

const footerCols = [
  {
    title: "Product",
    links: [
      { label: "Analyze Video", href: "/analyze" },
      { label: "Unified API", href: "/docs" },
      { label: "Mobile App", href: "/download" },
      { label: "About", href: "/about" },
    ],
  },
  {
    title: "Ecosystem",
    links: [
      { label: "Documentation", href: "/docs" },
      { label: "GitHub", href: "https://github.com/Aryanssingh6/veritas-ai" },
      { label: "Models", href: "/docs" },
      { label: "Releases", href: "/download" },
    ],
  },
  {
    title: "Community",
    links: [
      { label: "LinkedIn", href: "https://linkedin.com" },
      { label: "X (Twitter)", href: "https://twitter.com" },
      { label: "YouTube", href: "https://youtube.com" },
      { label: "GitHub Issues", href: "https://github.com/Aryanssingh6/veritas-ai/issues" },
    ],
  },
];

export default function Footer() {
  return (
    <footer style={{ backgroundColor: "#000", borderTop: "1px solid rgba(255,255,255,0.05)", padding: "80px 24px" }}>
      <div style={{ maxWidth: "1200px", margin: "0 auto", display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "60px" }}>
        <div>
          <Logo />
          <p style={{ fontSize: "0.95rem", color: "rgba(255,255,255,0.4)", marginTop: "20px", maxWidth: "300px", lineHeight: "1.7" }}>
            Open-source AI platform for advanced deepfake detection, temporal modeling, and frame-level spatial analysis.
          </p>
          <div style={{ marginTop: "32px", fontSize: "0.85rem", color: "rgba(255,255,255,0.25)" }}>
            © {new Date().getFullYear()} Veritas.ai. All rights reserved.
          </div>
        </div>

        <div style={{ display: "flex", gap: "80px", flexWrap: "wrap" }}>
          {footerCols.map((col) => (
            <div key={col.title}>
              <h4 style={{ fontSize: "0.8rem", fontWeight: "700", color: "rgba(255,255,255,0.5)", marginBottom: "20px", textTransform: "uppercase", letterSpacing: "0.1em" }}>{col.title}</h4>
              <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
                {col.links.map((l) => (
                  <Link
                    key={l.label}
                    href={l.href}
                    target={l.href.startsWith("http") ? "_blank" : undefined}
                    rel={l.href.startsWith("http") ? "noopener noreferrer" : undefined}
                    style={{ fontSize: "0.9rem", color: "rgba(255,255,255,0.45)", textDecoration: "none", transition: "color 0.2s ease" }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = "#fff")}
                    onMouseLeave={(e) => (e.currentTarget.style.color = "rgba(255,255,255,0.45)")}
                  >
                    {l.label}
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </footer>
  );
}
