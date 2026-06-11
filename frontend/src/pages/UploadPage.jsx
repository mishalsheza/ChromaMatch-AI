import React, { useState, useRef } from "react";
import {
  Camera,
  Image as ImageIcon,
  Sparkles,
  RefreshCw,
  AlertCircle,
  Move,
  Target,
} from "lucide-react";
import DropZone from "../components/DropZone";
import DraggableColorSampler from "../components/DraggableColorSampler";

export default function UploadPage({ onUploadComplete, isLoading, error }) {
  const [useWebcam, setUseWebcam] = useState(false);
  const [stream, setStream] = useState(null);
  const [countdown, setCountdown] = useState(null);
  const [camError, setCamError] = useState(null);
  const [showManualSampler, setShowManualSampler] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [useManualMode, setUseManualMode] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const videoRef = useRef(null);

  // Manage object URL lifecycle
  React.useEffect(() => {
    if (showManualSampler && uploadedImage) {
      const url = URL.createObjectURL(uploadedImage);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    }
    setPreviewUrl(null);
  }, [showManualSampler, uploadedImage]);

  // Handle file upload
  const handleFileUpload = (file) => {
    if (useManualMode) {
      setUploadedImage(file);
      setShowManualSampler(true);
    } else {
      onUploadComplete(file);
    }
  };

  // Handle samples collected from manual sampler
  const handleSamplesCollected = (samples) => {
    const formData = new FormData();
    formData.append("image", uploadedImage);
    formData.append("manual_sampling", "true");
    formData.append(
      "sampling_points",
      JSON.stringify(
        samples.map((s) => ({
          x: s.originalCoords.x,
          y: s.originalCoords.y,
          label: s.label,
        })),
      ),
    );

    onUploadComplete(uploadedImage, { manualSampling: true, samples });
    setShowManualSampler(false);
  };

  const startCamera = async () => {
    setCamError(null);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setUseWebcam(true);
    } catch (err) {
      console.error("Camera access failed:", err);
      setCamError(
        "Could not access camera. Please ensure permissions are granted or upload a file instead.",
      );
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    setStream(null);
    setUseWebcam(false);
    setCountdown(null);
  };

  const capturePhoto = () => {
    if (countdown !== null) return;

    let count = 3;
    setCountdown(count);

    const interval = setInterval(() => {
      count -= 1;
      if (count === 0) {
        clearInterval(interval);
        setCountdown(null);
        executeCapture();
      } else {
        setCountdown(count);
      }
    }, 800);
  };

  const executeCapture = () => {
    if (!videoRef.current) return;

    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], "selfie_capture.png", {
          type: "image/png",
        });
        if (useManualMode) {
          setUploadedImage(file);
          setShowManualSampler(true);
          stopCamera();
        } else {
          onUploadComplete(file);
          stopCamera();
        }
      }
    }, "image/png");
  };

  // If showing manual sampler, render it
  // In UploadPage.jsx
  if (showManualSampler && uploadedImage) {
    return (
      <DraggableColorSampler
        imageUrl={previewUrl}
        onSamplesCollected={handleSamplesCollected}
        onCancel={() => {
          setShowManualSampler(false);
          setUploadedImage(null);
        }}
      />
    );
  }

  return (
    <div
      style={{
        maxWidth: "800px",
        margin: "40px auto",
        padding: "0 20px",
        textAlign: "center",
      }}
    >
      {/* Brand Introduction Splash */}
      <div style={{ marginBottom: "40px" }}>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            background: "var(--accent-gold-glow)",
            color: "var(--accent-gold)",
            padding: "6px 14px",
            borderRadius: "20px",
            fontSize: "0.8rem",
            fontWeight: "600",
            textTransform: "uppercase",
            letterSpacing: "0.1em",
            marginBottom: "16px",
            border: "1px solid rgba(212,175,55,0.2)",
          }}
        >
          <Sparkles size={12} />
          <span>AI-Powered Personal Color Analytics</span>
        </div>

        <h1
          className="luxury-title"
          style={{
            fontSize: "3rem",
            color: "#FFFFFF",
            marginBottom: "16px",
            lineHeight: "1.1",
          }}
        >
          ShadeSense <span style={{ color: "var(--accent-gold)" }}>AI</span>
        </h1>

        <p
          style={{
            fontSize: "1.1rem",
            color: "var(--text-secondary)",
            maxWidth: "520px",
            margin: "0 auto",
            lineHeight: "1.6",
            fontWeight: "300",
          }}
        >
          Discover your objective skin parameters, Individual Typology Angle
          (ITA), and customized luxury matching cosmetics in seconds.
        </p>
      </div>

      {/* Mode Selection Toggle */}
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: "16px",
          marginBottom: "24px",
          background: "rgba(0,0,0,0.3)",
          padding: "8px",
          borderRadius: "40px",
          width: "fit-content",
          margin: "0 auto 24px auto",
        }}
      >
        <button
          onClick={() => setUseManualMode(false)}
          style={{
            padding: "10px 24px",
            borderRadius: "32px",
            border: "none",
            background: !useManualMode ? "var(--accent-gold)" : "transparent",
            color: !useManualMode ? "#000" : "var(--text-secondary)",
            cursor: "pointer",
            fontWeight: "500",
            transition: "all 0.3s ease",
          }}
        >
          <Target size={16} style={{ display: "inline", marginRight: "8px" }} />
          Auto-Detect
        </button>
        <button
          onClick={() => setUseManualMode(true)}
          style={{
            padding: "10px 24px",
            borderRadius: "32px",
            border: "none",
            background: useManualMode ? "var(--accent-gold)" : "transparent",
            color: useManualMode ? "#000" : "var(--text-secondary)",
            cursor: "pointer",
            fontWeight: "500",
            transition: "all 0.3s ease",
          }}
        >
          <Move size={16} style={{ display: "inline", marginRight: "8px" }} />
          Manual Placement
        </button>
      </div>

      {/* Main Glass Panel Interface */}
      <div
        className="glass-panel"
        style={{
          padding: "32px",
          borderRadius: "24px",
          boxShadow: "var(--shadow-dark)",
          border: "1px solid rgba(255, 255, 255, 0.05)",
          background: "rgba(12, 12, 16, 0.7)",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {isLoading ? (
          <div
            style={{
              padding: "60px 20px",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <div
              style={{
                width: "64px",
                height: "64px",
                borderRadius: "50%",
                border: "3px solid rgba(212, 175, 55, 0.1)",
                borderTopColor: "var(--accent-gold)",
                animation: "shimmer 1.5s infinite linear",
                marginBottom: "24px",
              }}
            />
            <h3
              style={{
                fontSize: "1.25rem",
                marginBottom: "8px",
                color: "#fff",
                fontWeight: "500",
              }}
            >
              Analyzing Facial Landmarks...
            </h3>
            <p
              style={{
                color: "var(--text-secondary)",
                fontSize: "0.9rem",
                maxWidth: "320px",
                lineHeight: "1.4",
              }}
            >
              {useManualMode
                ? "Preparing manual sampling interface..."
                : "We are standardizing exposure levels, extracting cheek coordinates, and resolving CIELAB skin algorithms."}
            </p>
          </div>
        ) : useWebcam ? (
          <div
            style={{
              position: "relative",
              width: "100%",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <div
              style={{
                position: "relative",
                borderRadius: "16px",
                overflow: "hidden",
                border: "1px solid var(--border-light)",
                maxWidth: "640px",
                width: "100%",
                aspectRatio: "4/3",
                background: "#000",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <video
                ref={videoRef}
                autoPlay
                playsInline
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  transform: "scaleX(-1)",
                }}
              />

              <div
                className="pulse-border"
                style={{
                  position: "absolute",
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                  width: "45%",
                  height: "65%",
                  border: "2px dashed var(--accent-gold)",
                  borderRadius: "50% 50% 50% 50% / 60% 60% 40% 40%",
                  zIndex: 2,
                  pointerEvents: "none",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <span
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--accent-gold)",
                    background: "rgba(0,0,0,0.6)",
                    padding: "4px 8px",
                    borderRadius: "4px",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                  }}
                >
                  Position Face Here
                </span>
              </div>

              {countdown !== null && (
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    backgroundColor: "rgba(0, 0, 0, 0.4)",
                    backdropFilter: "blur(4px)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    zIndex: 3,
                  }}
                >
                  <span
                    style={{
                      fontSize: "7rem",
                      fontWeight: "bold",
                      color: "var(--accent-gold)",
                      filter: "drop-shadow(0 4px 12px rgba(0,0,0,0.5))",
                    }}
                  >
                    {countdown}
                  </span>
                </div>
              )}
            </div>

            <div style={{ display: "flex", gap: "12px", marginTop: "20px" }}>
              <button
                onClick={capturePhoto}
                className="btn-gold"
                disabled={countdown !== null}
              >
                <Camera size={18} />
                <span>Capture Match</span>
              </button>

              <button
                onClick={stopCamera}
                className="btn-outline"
                disabled={countdown !== null}
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "24px",
            }}
          >
            <DropZone onFileSelect={handleFileUpload} disabled={isLoading} />

            <div
              style={{
                display: "flex",
                alignItems: "center",
                width: "100%",
                margin: "8px 0",
              }}
            >
              <div
                style={{
                  flexGrow: 1,
                  height: "1px",
                  background: "var(--border-light)",
                }}
              />
              <span
                style={{
                  padding: "0 16px",
                  color: "var(--text-muted)",
                  fontSize: "0.8rem",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                or
              </span>
              <div
                style={{
                  flexGrow: 1,
                  height: "1px",
                  background: "var(--border-light)",
                }}
              />
            </div>

            <button
              onClick={startCamera}
              className="btn-outline"
              style={{ padding: "14px 32px" }}
            >
              <Camera size={18} />
              <span>Use Live Video Capture</span>
            </button>

            {useManualMode && (
              <div
                style={{
                  marginTop: "16px",
                  padding: "12px",
                  background: "rgba(212, 175, 55, 0.1)",
                  borderRadius: "12px",
                  border: "1px solid rgba(212, 175, 55, 0.2)",
                }}
              >
                <p
                  style={{
                    fontSize: "0.85rem",
                    color: "var(--accent-gold)",
                    margin: 0,
                  }}
                >
                  ✨ Manual Mode: After uploading, you'll be able to drag
                  sampling points to exact cheek positions
                </p>
              </div>
            )}
          </div>
        )}

        {(error || camError) && (
          <div
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: "10px",
              color: "#e76f51",
              fontSize: "0.88rem",
              marginTop: "24px",
              background: "rgba(231, 111, 81, 0.08)",
              padding: "14px",
              borderRadius: "12px",
              border: "1px solid rgba(231, 111, 81, 0.15)",
              textAlign: "left",
            }}
          >
            <AlertCircle
              size={20}
              style={{ flexShrink: 0, marginTop: "2px" }}
            />
            <div>
              <strong style={{ display: "block", marginBottom: "2px" }}>
                Analysis Encountered an Issue
              </strong>
              <span>{error || camError}</span>
            </div>
          </div>
        )}
      </div>

      {/* Trust factors footer */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "20px",
          marginTop: "40px",
          borderTop: "1px solid var(--border-light)",
          paddingTop: "32px",
        }}
      >
        <div>
          <h4
            style={{
              fontSize: "0.9rem",
              color: "#fff",
              marginBottom: "6px",
              fontWeight: "500",
            }}
          >
            🔬 Objective L\*a\*b\* Space
          </h4>
          <p
            style={{
              fontSize: "0.78rem",
              color: "var(--text-secondary)",
              lineHeight: "1.4",
            }}
          >
            Color science analysis guarantees accurate metrics independent of
            sensory biases.
          </p>
        </div>
        <div>
          <h4
            style={{
              fontSize: "0.9rem",
              color: "#fff",
              marginBottom: "6px",
              fontWeight: "500",
            }}
          >
            💡 Ambient Balance Correction
          </h4>
          <p
            style={{
              fontSize: "0.78rem",
              color: "var(--text-secondary)",
              lineHeight: "1.4",
            }}
          >
            Gray World scale white balancing and CLAHE normalizes indoor shadows
            and exposure levels.
          </p>
        </div>
        <div>
          <h4
            style={{
              fontSize: "0.9rem",
              color: "#fff",
              marginBottom: "6px",
              fontWeight: "500",
            }}
          >
            🎯{" "}
            {useManualMode
              ? "Precise Manual Sampling"
              : "AI Landmark Detection"}
          </h4>
          <p
            style={{
              fontSize: "0.78rem",
              color: "var(--text-secondary)",
              lineHeight: "1.4",
            }}
          >
            {useManualMode
              ? "Click and drag sampling points to exactly where you want to measure skin color."
              : "Real-time 3D facial mesh maps left and right cheek locations perfectly."}
          </p>
        </div>
      </div>
    </div>
  );
}
