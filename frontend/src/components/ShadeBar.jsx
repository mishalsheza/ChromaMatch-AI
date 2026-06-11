import React from 'react';

export default function ShadeBar({ L, skinType, cosmeticDepth, ITA }) {
  // L is CIELAB Lightness, usually between 0 (black) and 100 (white).
  // We can calculate the percentage position based on L.
  // Standard human skin ranges from about L=20 (deepest dark) to L=92 (fairest light).
  // Let's normalize it to [0, 100] for the visual positioning.
  const positionPercent = Math.min(Math.max(L, 0), 100);

  // Gradient representing human skin tone range from Fair to Deep
  const skinGradient = 'linear-gradient(to right, #FFF5EB 0%, #F5D6B8 25%, #D29E74 50%, #9B683C 75%, #4A2E16 100%)';

  return (
    <div style={{ margin: '24px 0', width: '100%' }}>
      <div 
        style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'baseline', 
          marginBottom: '10px' 
        }}
      >
        <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Skin Depth Spectrum
        </span>
        <span style={{ fontSize: '1rem', color: 'var(--accent-gold)', fontWeight: '600' }}>
          {cosmeticDepth} Tone ({skinType})
        </span>
      </div>

      {/* The Spectrum Bar Container */}
      <div 
        style={{ 
          height: '24px', 
          width: '100%', 
          borderRadius: '12px', 
          background: skinGradient,
          position: 'relative',
          boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.3), 0 2px 8px rgba(0,0,0,0.15)',
          border: '1px solid rgba(255,255,255,0.05)'
        }}
      >
        {/* The Glow Pointer */}
        <div 
          style={{ 
            position: 'absolute', 
            left: `${positionPercent}%`, 
            top: '50%',
            transform: 'translate(-50%, -50%)',
            transition: 'left 1s cubic-bezier(0.16, 1, 0.3, 1)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            zIndex: 2
          }}
        >
          {/* Main Pointer Ring */}
          <div 
            style={{ 
              width: '32px', 
              height: '32px', 
              borderRadius: '50%', 
              backgroundColor: 'transparent', 
              border: '3px solid #FFF', 
              boxShadow: '0 0 12px var(--accent-gold), 0 4px 10px rgba(0,0,0,0.5)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer'
            }}
          >
            <div 
              style={{ 
                width: '14px', 
                height: '14px', 
                borderRadius: '50%', 
                backgroundColor: 'var(--accent-gold)' 
              }} 
            />
          </div>
        </div>
      </div>

      {/* Scale Labels */}
      <div 
        style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          marginTop: '8px', 
          fontSize: '0.75rem', 
          color: 'var(--text-muted)',
          padding: '0 4px'
        }}
      >
        <span>Fair</span>
        <span>Light</span>
        <span>Medium</span>
        <span>Tan</span>
        <span>Deep</span>
      </div>

      {/* Scientific ITA Details */}
      <div 
        className="glass-panel" 
        style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          padding: '12px 16px', 
          marginTop: '16px',
          borderRadius: '10px',
          border: '1px solid rgba(255, 255, 255, 0.04)',
          background: 'rgba(255, 255, 255, 0.01)'
        }}
      >
        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          Individual Typology Angle (ITA)
        </span>
        <span style={{ fontSize: '0.85rem', color: '#F4F4F5', fontWeight: '500', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ color: 'var(--accent-gold)', fontWeight: '600' }}>{ITA}°</span>
          <span style={{ color: 'var(--text-muted)' }}>|</span>
          <span>{skinType}</span>
        </span>
      </div>
    </div>
  );
}
