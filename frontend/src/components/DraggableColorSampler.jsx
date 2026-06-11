import React, { useState, useRef, useEffect, useCallback } from "react";

const DraggableColorSampler = ({ imageUrl, onSamplesCollected, onCancel }) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const samplesRef = useRef([]);
  const draggingIndexRef = useRef(null);
  const isDraggingRef = useRef(false);
  const [, forceUpdate] = useState(0);

  const POINT_RADIUS = 14;
  const LABELS = ["Left Cheek", "Right Cheek", "Point 3", "Point 4"];

  // ── Draw ────────────────────────────────────────────────────────────────────
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    samplesRef.current.forEach((s, idx) => {
      const dragging = draggingIndexRef.current === idx;

      // Crosshairs
      ctx.beginPath();
      ctx.moveTo(s.x - 22, s.y); ctx.lineTo(s.x - POINT_RADIUS + 2, s.y);
      ctx.moveTo(s.x + POINT_RADIUS - 2, s.y); ctx.lineTo(s.x + 22, s.y);
      ctx.moveTo(s.x, s.y - 22); ctx.lineTo(s.x, s.y - POINT_RADIUS + 2);
      ctx.moveTo(s.x, s.y + POINT_RADIUS - 2); ctx.lineTo(s.x, s.y + 22);
      ctx.strokeStyle = "#d4af37";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Outer ring
      ctx.beginPath();
      ctx.arc(s.x, s.y, POINT_RADIUS + (dragging ? 5 : 3), 0, 2 * Math.PI);
      ctx.strokeStyle = dragging ? "rgba(212,175,55,0.9)" : "rgba(212,175,55,0.5)";
      ctx.lineWidth = dragging ? 3 : 2;
      ctx.stroke();

      // Color fill
      ctx.beginPath();
      ctx.arc(s.x, s.y, POINT_RADIUS - 2, 0, 2 * Math.PI);
      ctx.fillStyle = s.color;
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Centre dot
      ctx.beginPath();
      ctx.arc(s.x, s.y, 3, 0, 2 * Math.PI);
      ctx.fillStyle = "#fff";
      ctx.fill();

      // Label badge
      const label = s.label;
      ctx.font = "bold 11px sans-serif";
      const tw = ctx.measureText(label).width;
      const bx = s.x - tw / 2 - 6;
      const by = s.y - POINT_RADIUS - 22;
      ctx.fillStyle = "rgba(0,0,0,0.75)";
      ctx.beginPath();
      ctx.roundRect(bx, by, tw + 12, 18, 4);
      ctx.fill();
      ctx.strokeStyle = "#d4af37";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.textAlign = "center";
      ctx.fillText(label, s.x, by + 13);
    });
  }, []);

  // ── Load image & set canvas size ────────────────────────────────────────────
  useEffect(() => {
    if (!imageUrl) return;
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.onload = () => {
      imageRef.current = img;
      const canvas = canvasRef.current;
      if (!canvas) return;

      const maxW = Math.min(600, img.width);
      canvas.width = maxW;
      canvas.height = Math.round(maxW * (img.height / img.width));

      // Place default points at cheek-ish positions
      const w = canvas.width, h = canvas.height;
      samplesRef.current = [
        { x: Math.round(w * 0.32), y: Math.round(h * 0.52), label: "Left Cheek",  color: sampleColor(Math.round(w * 0.32), Math.round(h * 0.52), img, w, h) },
        { x: Math.round(w * 0.68), y: Math.round(h * 0.52), label: "Right Cheek", color: sampleColor(Math.round(w * 0.68), Math.round(h * 0.52), img, w, h) },
      ];
      draw();
      forceUpdate(n => n + 1);
    };
    img.src = imageUrl;
  }, [imageUrl, draw]);

  // ── Colour sampling ─────────────────────────────────────────────────────────
  function sampleColor(cx, cy, img, canvasW, canvasH) {
    const tmp = document.createElement("canvas");
    tmp.width = 1; tmp.height = 1;
    const tctx = tmp.getContext("2d");
    const imgX = (cx / canvasW) * img.width;
    const imgY = (cy / canvasH) * img.height;
    tctx.drawImage(img, imgX, imgY, 1, 1, 0, 0, 1, 1);
    const [r, g, b] = tctx.getImageData(0, 0, 1, 1).data;
    return `#${r.toString(16).padStart(2,"0")}${g.toString(16).padStart(2,"0")}${b.toString(16).padStart(2,"0")}`;
  }

  // ── Canvas → client coords ───────────────────────────────────────────────────
  function toCanvas(clientX, clientY) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width  / rect.width;
    const scaleY = canvas.height / rect.height;
    const pad = POINT_RADIUS + 5;
    return {
      x: Math.max(pad, Math.min(canvas.width  - pad, (clientX - rect.left) * scaleX)),
      y: Math.max(pad, Math.min(canvas.height - pad, (clientY - rect.top)  * scaleY)),
    };
  }

  // ── Hit-test ─────────────────────────────────────────────────────────────────
  function hitTest(x, y) {
    for (let i = 0; i < samplesRef.current.length; i++) {
      const s = samplesRef.current[i];
      if (Math.hypot(s.x - x, s.y - y) < POINT_RADIUS + 10) return i;
    }
    return -1;
  }

  // ── Mouse handlers ───────────────────────────────────────────────────────────
  const onMouseDown = (e) => {
    const { x, y } = toCanvas(e.clientX, e.clientY);
    const idx = hitTest(x, y);
    if (idx !== -1) {
      draggingIndexRef.current = idx;
      isDraggingRef.current = false;
      e.preventDefault();
    }
  };

  const onMouseMove = useCallback((e) => {
    if (draggingIndexRef.current === null) return;
    isDraggingRef.current = true;
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;

    const { x, y } = toCanvas(e.clientX, e.clientY);
    const color = sampleColor(x, y, img, canvas.width, canvas.height);

    samplesRef.current = samplesRef.current.map((s, i) =>
      i === draggingIndexRef.current ? { ...s, x, y, color } : s
    );
    draw();
  }, [draw]);

  const onMouseUp = useCallback(() => {
    draggingIndexRef.current = null;
    draw();
    forceUpdate(n => n + 1);
  }, [draw]);

  // Attach move/up to window so drag works even outside canvas
  useEffect(() => {
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup",   onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup",   onMouseUp);
    };
  }, [onMouseMove, onMouseUp]);

  // ── Confirm ──────────────────────────────────────────────────────────────────
  const confirm = () => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;
    onSamplesCollected(
      samplesRef.current.map(s => ({
        ...s,
        originalCoords: {
          x: (s.x / canvas.width)  * img.width,
          y: (s.y / canvas.height) * img.height,
        },
      }))
    );
  };

  const samples = samplesRef.current;

  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:"16px" }}>
      <div style={{ textAlign:"center" }}>
        <h3 style={{ fontSize:"1.2rem", color:"#fff", margin:"0 0 6px" }}>
          Drag Points to Cheeks
        </h3>
        <p style={{ color:"#888", fontSize:"0.85rem", margin:0 }}>
          Click and drag the gold circles to the correct position on each cheek
        </p>
      </div>

      <canvas
        ref={canvasRef}
        onMouseDown={onMouseDown}
        style={{
          borderRadius:"12px",
          border:"2px solid #333",
          cursor: draggingIndexRef.current !== null ? "grabbing" : "grab",
          maxWidth:"100%",
          display:"block",
          backgroundColor:"#111",
        }}
      />

      {/* Colour swatches */}
      {samples.length > 0 && (
        <div style={{ display:"flex", gap:"16px" }}>
          {samples.map((s, i) => (
            <div key={i} style={{ textAlign:"center" }}>
              <div style={{
                width:44, height:44, borderRadius:8,
                background:s.color,
                border:"2px solid #555",
                margin:"0 auto 4px",
              }}/>
              <span style={{ fontSize:"0.7rem", color:"#888" }}>{s.label}</span><br/>
              <span style={{ fontSize:"0.65rem", color:"#555", fontFamily:"monospace" }}>{s.color}</span>
            </div>
          ))}
        </div>
      )}

      <div style={{ display:"flex", gap:"12px" }}>
        <button onClick={confirm} style={{
          background:"#d4af37", color:"#000", border:"none",
          padding:"10px 28px", borderRadius:"8px",
          fontWeight:"bold", cursor:"pointer", fontSize:"0.9rem",
        }}>
          Analyze
        </button>
        <button onClick={onCancel} style={{
          background:"transparent", color:"#fff",
          border:"1px solid #555", padding:"10px 20px",
          borderRadius:"8px", cursor:"pointer", fontSize:"0.9rem",
        }}>
          Cancel
        </button>
      </div>
    </div>
  );
};

export default DraggableColorSampler;