import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, AlertTriangle } from 'lucide-react';

export default function DropZone({ onFileSelect, disabled }) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (disabled) return;
    
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const validateAndProcessFile = (file) => {
    if (!file) return;
    
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
      setError("Please upload a valid image file (PNG, JPG, or WEBP).");
      return;
    }
    
    setError(null);
    onFileSelect(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e) => {
    if (disabled) return;
    if (e.target.files && e.target.files[0]) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    if (disabled) return;
    fileInputRef.current.click();
  };

  return (
    <div className="dropzone-container" style={{ width: '100%' }}>
      <div
        className={`glass-panel dropzone-area ${isDragActive ? 'active' : ''} ${disabled ? 'disabled' : ''}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={onButtonClick}
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '40px 24px',
          border: `2px dashed ${isDragActive ? 'var(--accent-gold)' : 'var(--border-light)'}`,
          cursor: disabled ? 'not-allowed' : 'pointer',
          borderRadius: '16px',
          transition: 'all 0.3s ease',
          backgroundColor: isDragActive ? 'rgba(212, 175, 55, 0.04)' : 'rgba(18, 18, 22, 0.45)',
          position: 'relative',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.webp"
          onChange={handleInputChange}
          style={{ display: 'none' }}
          disabled={disabled}
        />

        <div 
          className="upload-icon-wrapper" 
          style={{ 
            width: '64px', 
            height: '64px', 
            borderRadius: '50%', 
            background: 'rgba(212, 175, 55, 0.05)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: '16px',
            border: '1px solid rgba(212, 175, 55, 0.1)',
            color: isDragActive ? 'var(--accent-gold-hover)' : 'var(--accent-gold)',
            transition: 'all 0.3s ease'
          }}
        >
          {isDragActive ? (
            <ImageIcon size={28} className="pulse-border" style={{ borderRadius: '50%', padding: '2px' }} />
          ) : (
            <UploadCloud size={28} />
          )}
        </div>

        <h3 style={{ fontSize: '1.2rem', marginBottom: '8px', color: 'var(--text-primary)', fontWeight: '500' }}>
          Drag & Drop your selfie
        </h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', textAlign: 'center', maxWidth: '280px', lineHeight: '1.4' }}>
          or <span style={{ color: 'var(--accent-gold)', fontWeight: '500' }}>browse files</span> from your device
        </p>

        <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '20px' }}>
          Supports PNG, JPG, JPEG, or WEBP (Max 10MB)
        </span>
      </div>

      {error && (
        <div 
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px', 
            color: '#e76f51', 
            fontSize: '0.85rem', 
            marginTop: '12px',
            background: 'rgba(231, 111, 81, 0.08)',
            padding: '10px 14px',
            borderRadius: '8px',
            border: '1px solid rgba(231, 111, 81, 0.15)'
          }}
        >
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
