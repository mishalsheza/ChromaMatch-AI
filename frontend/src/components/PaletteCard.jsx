import React, { useState } from 'react';
import { Clipboard, Check } from 'lucide-react';

function SwatchItem({ name, hex, description }) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(hex);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div 
      className="glass-panel hover-lift"
      onClick={copyToClipboard}
      style={{
        display: 'flex',
        alignItems: 'center',
        padding: '12px',
        borderRadius: '12px',
        cursor: 'pointer',
        border: '1px solid rgba(255, 255, 255, 0.05)',
        background: 'rgba(18, 18, 22, 0.4)',
        gap: '12px',
        position: 'relative',
        transition: 'all 0.3s ease'
      }}
    >
      {/* Color Circle Swatch */}
      <div 
        style={{
          width: '42px',
          height: '42px',
          borderRadius: '50%',
          backgroundColor: hex,
          border: '2px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 4px 10px rgba(0,0,0,0.25)',
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        {copied && <Check size={16} style={{ color: '#fff', filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.5))' }} />}
      </div>

      <div style={{ flexGrow: 1, minWidth: 0 }}>
        <h4 style={{ fontSize: '0.88rem', color: '#f4f4f5', fontWeight: '600', marginBottom: '2px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {name}
        </h4>
        <span style={{ fontSize: '0.75rem', color: 'var(--accent-gold)', fontFamily: 'monospace' }}>
          {hex}
        </span>
        {description && (
          <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: '1.2' }}>
            {description}
          </p>
        )}
      </div>

      <div style={{ color: copied ? 'var(--accent-gold)' : 'var(--text-muted)', flexShrink: 0 }}>
        {copied ? <span style={{ fontSize: '0.7rem', fontWeight: '500', color: 'var(--accent-gold)' }}>Copied!</span> : <Clipboard size={14} />}
      </div>
    </div>
  );
}

export default function PaletteCard({ title, swatches, isClothing = false }) {
  const [activeTab, setActiveTab] = useState(isClothing ? 'clothing' : Object.keys(swatches)[0]);

  return (
    <div className="glass-panel" style={{ padding: '20px', borderRadius: '16px', height: '100%' }}>
      <h3 style={{ fontSize: '1.25rem', marginBottom: '16px', borderBottom: '1px solid var(--border-light)', paddingBottom: '10px', color: '#fff', fontWeight: '500', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>{title}</span>
        {isClothing && <span style={{ fontSize: '0.75rem', color: 'var(--accent-gold)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Wardrobe Guide</span>}
      </h3>

      {/* Categories Tabs if makeup */}
      {!isClothing && (
        <div style={{ display: 'flex', gap: '6px', marginBottom: '16px', background: 'rgba(0,0,0,0.2)', padding: '4px', borderRadius: '8px' }}>
          {Object.keys(swatches).map((category) => (
            <button
              key={category}
              onClick={() => setActiveTab(category)}
              style={{
                flexGrow: 1,
                padding: '8px 12px',
                borderRadius: '6px',
                border: 'none',
                background: activeTab === category ? 'rgba(212, 175, 55, 0.1)' : 'transparent',
                color: activeTab === category ? 'var(--accent-gold)' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontSize: '0.8rem',
                fontWeight: '600',
                textTransform: 'capitalize',
                transition: 'all 0.3s ease'
              }}
            >
              {category}
            </button>
          ))}
        </div>
      )}

      {/* Grid of swatches */}
      <div 
        style={{ 
          display: 'grid', 
          gridTemplateColumns: '1fr',
          gap: '10px',
          maxHeight: '320px',
          overflowY: 'auto',
          paddingRight: '4px'
        }}
      >
        // Around line 121, replace the swatches.map line with:

{swatches && Array.isArray(swatches) ? (
  swatches.map((item, idx) => (
    <SwatchItem key={idx} name={item.name} hex={item.hex} description={item.description} />
  ))
) : swatches && swatches[activeTab] && Array.isArray(swatches[activeTab]) ? (
  swatches[activeTab].map((item, idx) => (
    <SwatchItem key={idx} name={item.name} hex={item.hex} />
  ))
) : (
  <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '20px' }}>
    No swatches available
  </div>
)}
      </div>
    </div>
  );
}
