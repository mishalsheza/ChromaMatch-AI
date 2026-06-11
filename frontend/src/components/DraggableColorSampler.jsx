import React, { useState, useRef, useEffect } from "react";
import { Move, Target, Check, X } from "lucide-react";

const DraggableColorSampler = ({ imageUrl, onSamplesCollected, onCancel }) => {
  const canvasRef = useRef(null);
  const [samples, setSamples] = useState([]);
  const [draggedPoint, setDraggedPoint] = useState(null);
  const [hoverPoint, setHoverPoint] = useState(null);
  const [imageDimensions, setImageDimensions] = useState({
    width: 0,
    height: 0,
  });
  const [displayDimensions, setDisplayDimensions] = useState({
    width: 0,
    height: 0,
  });
  const [loadedImg, setLoadedImg] = useState(null);
  const [showHelper, setShowHelper] = useState(true);
  const [isDragging, setIsDragging] = useState(false);

  const REQUIRED_POINTS = 2;
  const POINT_RADIUS = 14;

  // Load image once
  useEffect(() => {
    if (!imageUrl) return;
    const img = new Image();
    img.onload = () => {
      setImageDimensions({ width: img.width, height: img.height });

      const maxWidth = Math.min(700, window.innerWidth - 100);
      const aspectRatio = img.height / img.width;
      const displayWidth = maxWidth;
      const displayHeight = displayWidth * aspectRatio;

      setDisplayDimensions({ width: displayWidth, height: displayHeight });
      setLoadedImg(img);
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // Draw canvas
  useEffect(() => {
    if (!loadedImg || !canvasRef.current) return;

    const canvas = canvasRef.current;
    canvas.width = displayDimensions.width;
    canvas.height = displayDimensions.height;
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(loadedImg, 0, 0, canvas.width, canvas.height);

    samples.forEach((sample, idx) => {
      const isHovered = hoverPoint === idx;

      // Outer glow
      ctx.beginPath();
      ctx.arc(
        sample.x,
        sample.y,
        POINT_RADIUS + (isHovered ? 6 : 4),
        0,
        2 * Math.PI,
      );
      ctx.strokeStyle = `rgba(212, 175, 55, ${isHovered ? 0.8 : 0.4})`;
      ctx.lineWidth = isHovered ? 3 : 2;
      ctx.stroke();

      // Main ring
      ctx.beginPath();
      ctx.arc(sample.x, sample.y, POINT_RADIUS, 0, 2 * Math.PI);
      ctx.strokeStyle = "#d4af37";
      ctx.lineWidth = 2.5;
      ctx.stroke();

      // Inner fill
      ctx.beginPath();
      ctx.arc(sample.x, sample.y, 8, 0, 2 * Math.PI);
      ctx.fillStyle = sample.color;
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Crosshairs
      ctx.beginPath();
      ctx.moveTo(sample.x - POINT_RADIUS - 6, sample.y);
      ctx.lineTo(sample.x - POINT_RADIUS + 2, sample.y);
      ctx.moveTo(sample.x + POINT_RADIUS - 2, sample.y);
      ctx.lineTo(sample.x + POINT_RADIUS + 6, sample.y);
      ctx.moveTo(sample.x, sample.y - POINT_RADIUS - 6);
      ctx.lineTo(sample.x, sample.y - POINT_RADIUS + 2);
      ctx.moveTo(sample.x, sample.y + POINT_RADIUS - 2);
      ctx.lineTo(sample.x, sample.y + POINT_RADIUS + 6);
      ctx.stroke();

      // Label
      const label = sample.label;
      ctx.font = 'bold 12px "Plus Jakarta Sans", sans-serif';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = "rgba(10, 10, 12, 0.95)";
      ctx.fillRect(
        sample.x - textWidth / 2 - 10,
        sample.y - POINT_RADIUS - 28,
        textWidth + 20,
        24,
      );
      ctx.strokeStyle = "rgba(212, 175, 55, 0.5)";
      ctx.strokeRect(
        sample.x - textWidth / 2 - 10,
        sample.y - POINT_RADIUS - 28,
        textWidth + 20,
        24,
      );

      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "center";
      ctx.fillText(label, sample.x, sample.y - POINT_RADIUS - 14);
    });
  }, [loadedImg, samples, hoverPoint, displayDimensions]);

  const getColorAtPoint = (canvas, x, y) => {
    const ctx = canvas.getContext("2d");
    const pixel = ctx.getImageData(x, y, 1, 1).data;
    return `#${((1 << 24) + (pixel[0] << 16) + (pixel[1] << 8) + pixel[2]).toString(16).slice(1)}`;
  };

  const getCanvasCoordinates = (clientX, clientY) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;

    let x = (clientX - rect.left) * scaleX;
    let y = (clientY - rect.top) * scaleY;

    // Constrain to canvas bounds
    x = Math.max(
      POINT_RADIUS + 5,
      Math.min(canvasRef.current.width - POINT_RADIUS - 5, x),
    );
    y = Math.max(
      POINT_RADIUS + 5,
      Math.min(canvasRef.current.height - POINT_RADIUS - 5, y),
    );

    return { x, y };
  };

  const handleCanvasClick = (e) => {
    // Don't add points while dragging
    if (isDragging || draggedPoint !== null) return;
    if (samples.length >= 4) return;

    const { x, y } = getCanvasCoordinates(e.clientX, e.clientY);
    const color = getColorAtPoint(canvasRef.current, x, y);

    let label = "";
    if (samples.length === 0) label = "Left Cheek";
    else if (samples.length === 1) label = "Right Cheek";
    else label = `Point ${samples.length + 1}`;

    const newSample = {
      x,
      y,
      color,
      label,
      originalCoords: {
        x: (x / displayDimensions.width) * imageDimensions.width,
        y: (y / displayDimensions.height) * imageDimensions.height,
      },
    };

    setSamples([...samples, newSample]);
    if (showHelper && samples.length + 1 >= REQUIRED_POINTS) {
      setTimeout(() => setShowHelper(false), 3000);
    }
  };

  const handleMouseMove = (e) => {
    if (!canvasRef.current) return;

    const { x: mouseX, y: mouseY } = getCanvasCoordinates(e.clientX, e.clientY);

    // Check hover
    const hoverIdx = samples.findIndex(
      (sample) =>
        Math.hypot(sample.x - mouseX, sample.y - mouseY) < POINT_RADIUS,
    );
    setHoverPoint(hoverIdx !== -1 ? hoverIdx : null);

    // Handle dragging
    if (draggedPoint !== null) {
      setIsDragging(true);
      const color = getColorAtPoint(canvasRef.current, mouseX, mouseY);

      const updatedSamples = [...samples];
      updatedSamples[draggedPoint] = {
        ...updatedSamples[draggedPoint],
        x: mouseX,
        y: mouseY,
        color: color,
        originalCoords: {
          x: (mouseX / displayDimensions.width) * imageDimensions.width,
          y: (mouseY / displayDimensions.height) * imageDimensions.height,
        },
      };
      setSamples(updatedSamples);
    }
  };

  const handleMouseDown = (e) => {
    if (hoverPoint !== null) {
      setDraggedPoint(hoverPoint);
      e.preventDefault();
      e.stopPropagation();
    }
  };

  const handleMouseUp = () => {
    setDraggedPoint(null);
    setIsDragging(false);
  };

  const removePoint = (index) => {
    if (samples.length <= REQUIRED_POINTS) return;
    const newSamples = samples.filter((_, i) => i !== index);
    setSamples(newSamples);
    setHoverPoint(null);
  };

  const confirmSamples = () => {
    if (samples.length >= REQUIRED_POINTS) {
      onSamplesCollected(samples);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "20px",
      }}
    >
      <div style={{ textAlign: "center", marginBottom: "8px" }}>
        <h3
          style={{
            fontSize: "1.3rem",
            color: "#fff",
            marginBottom: "8px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "10px",
          }}
        >
          <Move size={22} style={{ color: "var(--accent-gold)" }} />
          Drag Sampling Points to Cheeks
        </h3>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
          {samples.length === 0 &&
            "📍 Click on your LEFT cheek to add first sampling point"}
          {samples.length === 1 &&
            "📍 Click on your RIGHT cheek to add second sampling point"}
          {samples.length >= REQUIRED_POINTS &&
            samples.length < 4 &&
            "✓ Click to add optional reference points (drag to adjust)"}
          {samples.length >= 4 && "✓ Maximum points reached"}
        </p>
      </div>

      <div style={{ position: "relative" }}>
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{
            borderRadius: "16px",
            border: "2px solid var(--border-light)",
            cursor:
              draggedPoint !== null
                ? "grabbing"
                : hoverPoint !== null
                  ? "grab"
                  : "crosshair",
            maxWidth: "100%",
            height: "auto",
            boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
            display: "block",
          }}
        />

        {showHelper && samples.length < REQUIRED_POINTS && (
          <div
            style={{
              position: "absolute",
              bottom: "20px",
              left: "50%",
              transform: "translateX(-50%)",
              background: "rgba(0,0,0,0.8)",
              padding: "8px 16px",
              borderRadius: "8px",
              color: "var(--accent-gold)",
              fontSize: "0.85rem",
              backdropFilter: "blur(8px)",
              pointerEvents: "none",
              whiteSpace: "nowrap",
            }}
          >
            💡 Click on cheeks to add sampling points
          </div>
        )}
      </div>

      {samples.length > 0 && (
        <div
          style={{
            display: "flex",
            gap: "12px",
            padding: "16px",
            background: "rgba(0,0,0,0.3)",
            borderRadius: "12px",
            flexWrap: "wrap",
            justifyContent: "center",
          }}
        >
          {samples.map((sample, idx) => (
            <div
              key={idx}
              style={{
                textAlign: "center",
                position: "relative",
                opacity: draggedPoint === idx ? 0.7 : 1,
              }}
            >
              <div
                style={{
                  width: "50px",
                  height: "50px",
                  borderRadius: "8px",
                  background: sample.color,
                  border:
                    hoverPoint === idx
                      ? "2px solid var(--accent-gold)"
                      : "1px solid rgba(255,255,255,0.2)",
                  marginBottom: "6px",
                  cursor:
                    samples.length > REQUIRED_POINTS ? "pointer" : "default",
                }}
                onClick={() =>
                  samples.length > REQUIRED_POINTS && removePoint(idx)
                }
              />
              <span
                style={{
                  fontSize: "0.7rem",
                  color: "var(--text-muted)",
                  display: "block",
                }}
              >
                {sample.label}
              </span>
              {samples.length > REQUIRED_POINTS && hoverPoint === idx && (
                <div
                  style={{
                    position: "absolute",
                    top: "-8px",
                    right: "-8px",
                    background: "#e76f51",
                    borderRadius: "50%",
                    width: "20px",
                    height: "20px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    cursor: "pointer",
                  }}
                >
                  <X size={12} />
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <div style={{ display: "flex", gap: "16px", marginTop: "8px" }}>
        <button
          onClick={confirmSamples}
          disabled={samples.length < REQUIRED_POINTS}
          className="btn-gold"
          style={{
            opacity: samples.length < REQUIRED_POINTS ? 0.5 : 1,
            padding: "12px 28px",
          }}
        >
          <Check size={18} />
          <span>Analyze Selected Points</span>
        </button>

        <button
          onClick={onCancel}
          className="btn-outline"
          style={{ padding: "12px 28px" }}
        >
          Cancel
        </button>
      </div>
    </div>
  );
};

export default DraggableColorSampler;
