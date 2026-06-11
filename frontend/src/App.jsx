import React, { useState } from 'react';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import { analyzeImage } from './services/api';
import { Sparkles } from 'lucide-react';

export default function App() {
  const [page, setPage] = useState('upload'); // 'upload' or 'results'
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [error, setError] = useState(null);

  const handleUploadComplete = async (file, options = {}) => {
    setIsLoading(true);
    setError(null);
    setImageFile(file);

    try {
      const response = await analyzeImage(file, options);
      if (response && response.success) {
        setResult(response);
        setPage('results');
      } else {
        throw new Error(response.error || "Failed to analyze skin tones.");
      }
    } catch (err) {
      console.error("Upload handler error:", err);
      setError(err.message || "An unexpected error occurred while communicating with the analysis server. Please check your connection and try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRestart = () => {
    setResult(null);
    setImageFile(null);
    setError(null);
    setPage('upload');
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      
      {/* Luxury Brand Header */}
      <header 
        style={{ 
          padding: '20px 40px', 
          borderBottom: '1px solid var(--border-light)',
          background: 'rgba(10, 10, 12, 0.4)',
          backdropFilter: 'blur(12px)',
          position: 'sticky',
          top: 0,
          zIndex: 100,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}
      >
        <div 
          onClick={handleRestart}
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '10px', 
            cursor: 'pointer',
            fontFamily: 'var(--font-serif)',
            fontSize: '1.4rem',
            color: '#FFFFFF',
            fontWeight: '600',
            letterSpacing: '0.02em'
          }}
        >
          <span style={{ color: 'var(--accent-gold)' }}>S</span>HADE<span style={{ color: 'var(--accent-gold)' }}>S</span>ENSE
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
          <Sparkles size={14} style={{ color: 'var(--accent-gold)' }} />
          <span style={{ fontWeight: '500' }}>Calibrated Spectrum matching v1.0</span>
        </div>
      </header>

      {/* Main Workspace Area */}
      <main style={{ flexGrow: 1, padding: '20px 0' }}>
        {page === 'upload' ? (
          <UploadPage 
            onUploadComplete={handleUploadComplete} 
            isLoading={isLoading} 
            error={error} 
          />
        ) : (
          <ResultsPage 
            result={result} 
            imageFile={imageFile} 
            onRestart={handleRestart} 
          />
        )}
      </main>

      {/* Footer */}
      <footer 
        style={{ 
          padding: '30px 20px', 
          textAlign: 'center', 
          borderTop: '1px solid var(--border-light)',
          background: '#060608',
          color: 'var(--text-muted)',
          fontSize: '0.78rem'
        }}
      >
        <p>© 2026 ShadeSense AI. Built using scientific L\*a\*b\* Color Space and MediaPipe Face Mesh matching algorithms. All rights reserved.</p>
      </footer>

    </div>
  );
}
