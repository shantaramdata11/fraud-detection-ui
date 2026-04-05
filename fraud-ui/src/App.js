import React, { useState } from 'react';
import './App.css';

function App() {
  const [inputData, setInputData] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  // YOUR LIVE AWS API GATEWAY URL
  const API_URL = "https://x1xjgf8i0i.execute-api.ap-south-1.amazonaws.com/default";

  const handleAnalyze = async () => {
    if (!inputData) return alert("Please enter transaction data");
    setLoading(true);
    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: inputData })
      });
      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error("Connection Error:", error);
      alert("Error connecting to API. Ensure your SageMaker Endpoint is InService.");
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '40px', textAlign: 'center', fontFamily: 'sans-serif', backgroundColor: '#f4f7f6', minHeight: '100vh' }}>
      <h1 style={{ color: '#232f3e' }}>🛡️ AWS Fraud Detection System</h1>
      <p>Powered by SageMaker XGBoost & AWS Lambda</p>
      
      <div style={{ margin: '20px auto', maxWidth: '700px', backgroundColor: 'white', border: '1px solid #ddd', padding: '30px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
        <h3>Transaction Features (V1-V28, Amount)</h3>
        <textarea 
          style={{ width: '100%', height: '120px', marginBottom: '20px', borderRadius: '8px', padding: '10px', border: '1px solid #ccc' }}
          placeholder="Paste comma-separated values here..."
          value={inputData}
          onChange={(e) => setInputData(e.target.value)}
        />
        <button 
          onClick={handleAnalyze}
          style={{ backgroundColor: '#ff9900', color: 'white', border: 'none', padding: '12px 25px', cursor: 'pointer', fontWeight: 'bold', borderRadius: '5px', fontSize: '16px' }}
        >
          {loading ? "PROCESSSING AI INFERENCE..." : "RUN AI DETECTION"}
        </button>

        {prediction && (
          <div style={{ marginTop: '30px', padding: '20px', borderRadius: '8px', borderLeft: '10px solid', borderLeftColor: prediction.prediction === 1 ? '#c53030' : '#166534', backgroundColor: prediction.prediction === 1 ? '#ffdce0' : '#dcfce7' }}>
            <h2 style={{ color: prediction.prediction === 1 ? '#c53030' : '#166534', margin: '0 0 10px 0' }}>
              {prediction.status}
            </h2>
            <p style={{ margin: 0 }}><strong>System Message:</strong> {prediction.message}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;