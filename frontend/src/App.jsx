import { useState } from "react";
import { predictEmail } from "./api";
import "./App.css"; 

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    setError("");
    setResult(null);

    const trimmed = text.trim();
    if (!trimmed) {
      setError("Please paste an email to analyze.");
      return;
    }

    try {
      setLoading(true);
      const res = await predictEmail(trimmed);
      if (res.error) {
        setError(res.error);
      } else {
        setResult(res);
      }
    } catch (err) {
      console.error(err);
      setError("Could not reach the server. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="card">
        <div className="header">
          <h1>AI Scam Shield</h1>
          <p>Advanced Hybrid-Model Detection</p>
        </div>

        <textarea
          className="email-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste email content here (Subject + Body)..."
          rows={6}
        />

        <button 
          className="analyze-btn" 
          onClick={handleAnalyze} 
          disabled={loading}
        >
          {loading ? "Analyzing Psychology & Context..." : "Analyze Email"}
        </button>

        {error && <div className="error-msg">{error}</div>}

        {result && !error && (
          <div className={`result-container ${result.is_scam ? "scam" : "safe"}`}>
            {/* Top Badge */}
            <div className="result-header">
              <div className="badge-container">
                <span className="label">VERDICT</span>
                <div className="result-badge">{result.prediction.toUpperCase()}</div>
              </div>
              <div className="confidence-container">
                <span className="label">RISK LEVEL</span>
                <div className="confidence-score">{result.confidence_percent}</div>
              </div>
            </div>

            <hr className="divider" />

            {/* Smart Triggers Section */}
            <div className="section-title">PSYCHOLOGICAL TRIGGERS</div>
            <div className="triggers-grid">
                <div className={`trigger-item ${result.triggers.urgency_count > 0 ? 'active' : ''}`}>
                    <span className="icon">⏰</span> Urgency ({result.triggers.urgency_count})
                </div>
                <div className={`trigger-item ${result.triggers.money_words > 0 ? 'active' : ''}`}>
                    <span className="icon">💸</span> Financial ({result.triggers.money_words})
                </div>
                <div className={`trigger-item ${result.triggers.generic_greeting ? 'active' : ''}`}>
                    <span className="icon">🤖</span> Generic Greeting
                </div>
            </div>

            {/* Models Comparison */}
            <div className="section-title" style={{marginTop: '1rem'}}>MODEL CONSENSUS</div>
            <div className="breakdown-grid">
              {/* TF-IDF Bar */}
              <div>
                <span className="progress-label">Keywords (TF-IDF)</span>
                <div className="progress-bg">
                  <div className="progress-fill" style={{ width: result.breakdown.tfidf_score, backgroundColor: "#3b82f6" }}></div>
                </div>
                <div style={{ textAlign: "right", fontSize: "0.8rem", color: "#a1a1aa", marginTop: "0.25rem" }}>
                  {result.breakdown.tfidf_score}
                </div>
              </div>

              {/* SBERT Bar */}
              <div>
                <span className="progress-label">Context (SBERT)</span>
                <div className="progress-bg">
                  <div className="progress-fill" style={{ width: result.breakdown.sbert_score, backgroundColor: "#a855f7" }}></div>
                </div>
                <div style={{ textAlign: "right", fontSize: "0.8rem", color: "#a1a1aa", marginTop: "0.25rem" }}>
                  {result.breakdown.sbert_score}
                </div>
              </div>
            </div>

            {/* LIME Visualization (NORMALIZED) */}
            <div className="section-title" style={{marginTop: '1.5rem'}}>
              KEYWORD IMPACT (Relative Importance)
            </div>
            <div className="lime-container">
                {(() => {
                  // 1. Calculate the total weight of all features returned to normalize them
                  const totalWeight = result.lime_explanation.reduce((sum, item) => sum + Math.abs(item[1]), 0);
                  
                  // Avoid divide by zero
                  const safeTotal = totalWeight || 1; 

                  return result.lime_explanation.map((item, index) => {
                    const word = item[0];
                    const rawWeight = item[1];
                    const isScamIndicator = rawWeight > 0;
                    
                    // 2. Calculate Normalized Percentage (Contribution to the decision)
                    // "Of the words that mattered, how much did THIS word matter?"
                    const relativeImpact = (Math.abs(rawWeight) / safeTotal) * 100;

                    return (
                        <div key={index} className="lime-row">
                            <span className="lime-word">"{word}"</span>
                            <div className="lime-bar-container">
                                {isScamIndicator ? (
                                    <div 
                                        className="lime-bar scam" 
                                        style={{ width: `${relativeImpact}%` }}
                                    ></div>
                                ) : (
                                    <div 
                                        className="lime-bar safe" 
                                        style={{ width: `${relativeImpact}%` }}
                                    ></div>
                                )}
                            </div>
                            <span className={`lime-score ${isScamIndicator ? 'text-red' : 'text-green'}`}>
                                {isScamIndicator ? "+" : ""}{relativeImpact.toFixed(1)}%
                            </span>
                        </div>
                    );
                  });
                })()}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;