import React, { useRef, useEffect, useState } from "react";

const CheekColorSampler = ({ imageUrl, onAnalyze, onCancel }) => {
  const canvasRef = useRef(null);
  const stateRef = useRef({
    img: null,
    markers: [
      { id: "left", label: "Left Cheek", nx: 0.32, ny: 0.52, color: "#888" },
      { id: "right", label: "Right Cheek", nx: 0.68, ny: 0.52, color: "#888" },
    ],
    dragging: null,
  });
  const [colors, setColors] = useState({ left: "#888", right: "#888" });
  const RADIUS = 16;

  // ── helpers ──────────────────────────────────────────────────────────────
  function sampleColor(img, nx, ny) {
    const t = document.createElement("canvas");
    t.width = t.height = 1;
    t.getContext("2d").drawImage(
      img,
      nx * img.width,
      ny * img.height,
      1,
      1,
      0,
      0,
      1,
      1,
    );
    const [r, g, b] = t.getContext("2d").getImageData(0, 0, 1, 1).data;
    return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
  }

  function draw() {
    const canvas = canvasRef.current;
    const { img, markers, dragging } = stateRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    markers.forEach((m) => {
      const x = m.nx * canvas.width,
        y = m.ny * canvas.height;
      const active = dragging === m.id;
      ctx.strokeStyle = "#d4af37";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x - 24, y);
      ctx.lineTo(x - RADIUS + 2, y);
      ctx.moveTo(x + RADIUS - 2, y);
      ctx.lineTo(x + 24, y);
      ctx.moveTo(x, y - 24);
      ctx.lineTo(x, y - RADIUS + 2);
      ctx.moveTo(x, y + RADIUS - 2);
      ctx.lineTo(x, y + 24);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, RADIUS + (active ? 5 : 3), 0, 2 * Math.PI);
      ctx.strokeStyle = active
        ? "rgba(212,175,55,0.9)"
        : "rgba(212,175,55,0.5)";
      ctx.lineWidth = active ? 3 : 2;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, RADIUS - 2, 0, 2 * Math.PI);
      ctx.fillStyle = m.color;
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      const tw = ctx.measureText(m.label).width;
      const bx = x - tw / 2 - 6,
        by = y - RADIUS - 24;
      ctx.fillStyle = "rgba(0,0,0,0.75)";
      ctx.beginPath();
      ctx.roundRect(bx, by, tw + 12, 18, 4);
      ctx.fill();
      ctx.strokeStyle = "#d4af37";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.fillText(m.label, x, by + 13);
    });
  }

  function toNorm(e) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const pad = RADIUS + 6;
    const cx = Math.max(
      pad,
      Math.min(
        canvas.width - pad,
        (e.clientX - rect.left) * (canvas.width / rect.width),
      ),
    );
    const cy = Math.max(
      pad,
      Math.min(
        canvas.height - pad,
        (e.clientY - rect.top) * (canvas.height / rect.height),
      ),
    );
    return { nx: cx / canvas.width, ny: cy / canvas.height };
  }

  function hitTest(e) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const cx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const cy = (e.clientY - rect.top) * (canvas.height / rect.height);
    for (const m of stateRef.current.markers) {
      if (
        Math.hypot(m.nx * canvas.width - cx, m.ny * canvas.height - cy) <
        RADIUS + 10
      )
        return m.id;
    }
    return null;
  }

  // ── load image ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!imageUrl) return;
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const maxW = Math.min(600, img.width);
      canvas.width = maxW;
      canvas.height = Math.round((maxW * img.height) / img.width);
      stateRef.current.img = img;
      stateRef.current.markers = stateRef.current.markers.map((m) => ({
        ...m,
        color: sampleColor(img, m.nx, m.ny),
      }));
      draw();
      setColors({
        left: stateRef.current.markers[0].color,
        right: stateRef.current.markers[1].color,
      });
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // ── NATIVE event listeners (the key fix) ──────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    function onDown(e) {
      const id = hitTest(e);
      if (id) {
        stateRef.current.dragging = id;
        e.preventDefault();
      }
    }

    function onMove(e) {
      if (!stateRef.current.dragging) return;
      const { img } = stateRef.current;
      const { nx, ny } = toNorm(e);
      const color = img ? sampleColor(img, nx, ny) : "#888";
      stateRef.current.markers = stateRef.current.markers.map((m) =>
        m.id === stateRef.current.dragging ? { ...m, nx, ny, color } : m,
      );
      draw();
      setColors({
        left: stateRef.current.markers[0].color,
        right: stateRef.current.markers[1].color,
      });
    }

    function onUp() {
      if (stateRef.current.dragging) {
        stateRef.current.dragging = null;
        draw();
      }
    }

    // native listeners — no React synthetic event layer
    canvas.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      canvas.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []); // empty deps — closures read stateRef.current, not stale state

  // ── confirm ───────────────────────────────────────────────────────────────
  const handleAnalyze = () => {
    const { img, markers } = stateRef.current;
    onAnalyze(
      markers.map((m) => ({
        label: m.label,
        x: m.nx,
        y: m.ny,
        color: m.color,
        originalCoords: {
          x: m.nx * (img?.width || 1),
          y: m.ny * (img?.height || 1),
        },
      })),
    );
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "16px",
      }}
    >
      <div style={{ textAlign: "center" }}>
        <h3 style={{ fontSize: "1.2rem", color: "#fff", margin: "0 0 6px" }}>
          Drag Points to Cheeks
        </h3>
        <p style={{ color: "#888", fontSize: "0.85rem", margin: 0 }}>
          Click and drag the gold circles to each cheek
        </p>
      </div>

      <canvas
        ref={canvasRef}
        style={{
          borderRadius: "12px",
          border: "2px solid #333",
          cursor: "grab",
          maxWidth: "100%",
          display: "block",
          background: "#111",
        }}
      />

      <div style={{ display: "flex", gap: "16px" }}>
        {[
          ["left", "Left Cheek"],
          ["right", "Right Cheek"],
        ].map(([id, label]) => (
          <div key={id} style={{ textAlign: "center" }}>
            <div
              style={{
                width: 44,
                height: 44,
                borderRadius: 8,
                background: colors[id],
                border: "2px solid #555",
                margin: "0 auto 4px",
              }}
            />
            <span style={{ fontSize: "0.7rem", color: "#888" }}>{label}</span>
            <br />
            <span
              style={{
                fontSize: "0.65rem",
                color: "#555",
                fontFamily: "monospace",
              }}
            >
              {colors[id]}
            </span>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: "12px" }}>
        <button
          onClick={handleAnalyze}
          style={{
            background: "#d4af37",
            color: "#000",
            border: "none",
            padding: "10px 28px",
            borderRadius: "8px",
            fontWeight: "bold",
            cursor: "pointer",
            fontSize: "0.9rem",
          }}
        >
          Analyze
        </button>
        <button
          onClick={onCancel}
          style={{
            background: "transparent",
            color: "#fff",
            border: "1px solid #555",
            padding: "10px 20px",
            borderRadius: "8px",
            cursor: "pointer",
            fontSize: "0.9rem",
          }}
        >
          Cancel
        </button>
      </div>
    </div>
  );
};

export default CheekColorSampler;
