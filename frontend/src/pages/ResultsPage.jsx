import React, { useEffect, useRef, useState } from 'react';
import { RefreshCw, Check, Sparkles, ShoppingBag, Eye, Copy } from 'lucide-react';
import ShadeBar from '../components/ShadeBar';
import PaletteCard from '../components/PaletteCard';

export default function ResultsPage({ result, imageFile, onRestart }) {
  const canvasRef = useRef(null);
  const [imageUrl, setImageUrl] = useState('');
  const [copiedFoundation, setCopiedFoundation] = useState(null);

  // 1. Generate local URL for the uploaded file
  useEffect(() => {
    if (imageFile) {
      const url = URL.createObjectURL(imageFile);
      setImageUrl(url);
      return () => URL.revokeObjectURL(url);
    }
  }, [imageFile]);

  // 2. Draw image & detected cheek landmarks on Canvas
  useEffect(() => {
    if (!imageUrl || !result) return;

    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      
      // Calculate display dimensions keeping aspect ratio
      const maxWidth = 500;
      const aspectRatio = img.height / img.width;
      const displayWidth = Math.min(maxWidth, img.width);
      const displayHeight = displayWidth * aspectRatio;

      canvas.width = displayWidth;
      canvas.height = displayHeight;

      // Draw background selfie
      ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

      // Scale coordinates from original image size to canvas size
      const scaleX = displayWidth / result.dimensions.width;
      const scaleY = displayHeight / result.dimensions.height;

      // Helper to draw a target crosshair
      const drawTarget = (x, y, label, hexColor) => {
        const cx = x * scaleX;
        const cy = y * scaleY;
        const radius = 16;

        // Outer glow circle
        ctx.beginPath();
        ctx.arc(cx, cy, radius + 4, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.lineWidth = 3;
        ctx.stroke();

        // Main Target Ring
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = '#D4AF37';
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // Inner solid core color of cheek
        ctx.beginPath();
        ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
        ctx.fillStyle = hexColor;
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();

        // Crosshairs
        ctx.beginPath();
        ctx.moveTo(cx - radius - 6, cy);
        ctx.lineTo(cx - radius + 2, cy);
        ctx.moveTo(cx + radius - 2, cy);
        ctx.lineTo(cx + radius + 6, cy);
        ctx.moveTo(cx, cy - radius - 6);
        ctx.lineTo(cx, cy - radius + 2);
        ctx.moveTo(cx, cy + radius - 2);
        ctx.lineTo(cx, cy + radius + 6);
        ctx.strokeStyle = '#D4AF37';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Label Tag Background
        const labelText = label;
        ctx.font = 'bold 11px -apple-system, system-ui, sans-serif';
        const textWidth = ctx.measureText(labelText).width;
        
        ctx.fillStyle = 'rgba(10, 10, 12, 0.85)';
        ctx.fillRect(cx - (textWidth / 2) - 8, cy - radius - 24, textWidth + 16, 18);
        ctx.strokeStyle = 'rgba(212, 175, 55, 0.4)';
        ctx.strokeRect(cx - (textWidth / 2) - 8, cy - radius - 24, textWidth + 16, 18);

        // Label Text
        ctx.fillStyle = '#FFFFFF';
        ctx.textAlign = 'center';
        ctx.fillText(labelText, cx, cy - radius - 11);
      };

      // Draw left & right cheek targets
      if (result.cheeks.left && result.cheeks.left.center) {
        drawTarget(
          result.cheeks.left.center[0], 
          result.cheeks.left.center[1], 
          "Left Cheek", 
          result.cheeks.left.hex
        );
      }
      if (result.cheeks.right && result.cheeks.right.center) {
        drawTarget(
          result.cheeks.right.center[0], 
          result.cheeks.right.center[1], 
          "Right Cheek", 
          result.cheeks.right.hex
        );
      }
    };
  }, [imageUrl, result]);

  const copyToClipboard = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedFoundation(idx);
    setTimeout(() => setCopiedFoundation(null), 2000);
  };

  const getUndertoneColor = () => {
    if (result.analysis.undertone === 'Warm') return '#E3C16F';
    if (result.analysis.undertone === 'Cool') return '#C8A2C8';
    return '#EAE6DF';
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '30px auto', padding: '0 20px' }}>
      
      {/* Top action bar */}
      <div 
        style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          marginBottom: '32px',
          borderBottom: '1px solid var(--border-light)',
          paddingBottom: '20px'
        }}
      >
        <div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Analysis Results</span>
          <h2 style={{ fontSize: '1.75rem', color: '#fff', fontWeight: '500' }}>Your Skin Profile Dashboard</h2>
        </div>
        <button onClick={onRestart} className="btn-outline" style={{ padding: '10px 20px', fontSize: '0.85rem' }}>
          <RefreshCw size={14} />
          <span>New Analysis</span>
        </button>
      </div>

      {/* Main Grid Layout */}
      <div 
        style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', 
          gap: '24px',
          marginBottom: '32px' 
        }}
      >
        
        {/* Left Card: Face mesh visualization */}
        <div 
          className="glass-panel" 
          style={{ 
            padding: '24px', 
            borderRadius: '20px', 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(12, 12, 16, 0.45)'
          }}
        >
          <h3 style={{ alignSelf: 'flex-start', fontSize: '1.15rem', color: '#fff', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '500' }}>
            <Eye size={18} style={{ color: 'var(--accent-gold)' }} />
            Cheek Pixel Localization
          </h3>
          
          <div 
            style={{ 
              borderRadius: '16px', 
              overflow: 'hidden', 
              border: '1px solid var(--border-light)',
              boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
              position: 'relative',
              maxWidth: '100%',
              background: '#0c0c0f'
            }}
          >
            <canvas ref={canvasRef} style={{ display: 'block', maxWidth: '100%' }} />
          </div>

          {/* Color swatch info */}
          <div 
            style={{ 
              display: 'grid', 
              gridTemplateColumns: '1fr 1fr', 
              gap: '12px', 
              width: '100%', 
              marginTop: '20px' 
            }}
          >
            {/* Left Cheek Swatch */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.02)' }}>
              <div style={{ width: '24px', height: '24px', borderRadius: '4px', background: result.cheeks.left.hex, border: '1px solid rgba(255,255,255,0.1)' }} />
              <div>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', display: 'block', textTransform: 'uppercase' }}>Left Color</span>
                <span style={{ fontSize: '0.8rem', color: '#f4f4f5', fontFamily: 'monospace' }}>{result.cheeks.left.hex}</span>
              </div>
            </div>
            
            {/* Right Cheek Swatch */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.02)' }}>
              <div style={{ width: '24px', height: '24px', borderRadius: '4px', background: result.cheeks.right.hex, border: '1px solid rgba(255,255,255,0.1)' }} />
              <div>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', display: 'block', textTransform: 'uppercase' }}>Right Color</span>
                <span style={{ fontSize: '0.8rem', color: '#f4f4f5', fontFamily: 'monospace' }}>{result.cheeks.right.hex}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Card: Objective skin tone parameters and foundations */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          {/* Undertone Summary Card */}
          <div 
            className="glass-panel" 
            style={{ 
              padding: '24px', 
              borderRadius: '20px', 
              borderLeft: `4px solid ${getUndertoneColor()}`,
              background: 'rgba(12, 12, 16, 0.45)'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Undertone</span>
                <h3 style={{ fontSize: '1.85rem', color: '#fff', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '500', marginTop: '2px' }}>
                  {result.analysis.undertone}
                  <span style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: getUndertoneColor() }} />
                </h3>
              </div>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: 'var(--accent-gold)', fontSize: '0.8rem', background: 'var(--accent-gold-glow)', padding: '4px 10px', borderRadius: '12px', border: '1px solid rgba(212,175,55,0.1)' }}>
                <Sparkles size={12} />
                <span>Calibrated Match</span>
              </div>
            </div>
            
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', lineHeight: '1.5' }}>
              {result.recommendations.seasonal_profile.description}
            </p>
          </div>

          {/* Skin Shade bar spectrum */}
          <div className="glass-panel" style={{ padding: '24px', borderRadius: '20px', background: 'rgba(12, 12, 16, 0.45)' }}>
            <ShadeBar 
              L={result.analysis.lab.L}
              skinType={result.analysis.skin_type}
              cosmeticDepth={result.analysis.cosmetic_depth}
              ITA={result.analysis.ita}
            />
          </div>

          {/* Foundation Brand matches */}
          <div className="glass-panel" style={{ padding: '24px', borderRadius: '20px', background: 'rgba(12, 12, 16, 0.45)' }}>
            <h3 style={{ fontSize: '1.15rem', color: '#fff', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '500' }}>
              <ShoppingBag size={18} style={{ color: 'var(--accent-gold)' }} />
              Luxury Foundation Matches
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {result.recommendations.foundations.map((item, idx) => (
                <div 
                  key={idx}
                  onClick={() => copyToClipboard(item.shade, idx)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '12px 16px',
                    borderRadius: '12px',
                    background: 'rgba(255, 255, 255, 0.02)',
                    border: '1px solid rgba(255, 255, 255, 0.04)',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease'
                  }}
                  className="hover-lift"
                >
                  <div>
                    <strong style={{ fontSize: '0.85rem', color: '#fff', display: 'block' }}>{item.brand}</strong>
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>{item.product}</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '0.85rem', color: 'var(--accent-gold)', fontWeight: '600', background: 'rgba(212, 175, 55, 0.05)', padding: '6px 10px', borderRadius: '6px', border: '1px solid rgba(212, 175, 55, 0.1)' }}>
                      {item.shade}
                    </span>
                    {copiedFoundation === idx ? (
                      <Check size={14} style={{ color: 'var(--accent-gold)' }} />
                    ) : (
                      <Copy size={12} style={{ color: 'var(--text-muted)' }} />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>
      </div>

      {/* Seasonal & Wardrobe + Makeup color swatches */}
      <div 
        style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', 
          gap: '24px',
          marginTop: '32px'
        }}
      >
        
        {/* Makeup recommendations */}
        <PaletteCard 
          title="Curated Cosmetic Palettes" 
          swatches={result.recommendations.makeup_swatches} 
          isClothing={false} 
        />
        
        {/* Wardrobe clothing guide */}
        <PaletteCard 
          title={`Seasonal Palette: ${result.recommendations.seasonal_profile.name}`} 
          swatches={result.recommendations.clothing_palette} 
          isClothing={true} 
        />
        
      </div>

    </div>
  );
}
