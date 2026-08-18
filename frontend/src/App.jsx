import React, { useState } from 'react';
import axios from 'axios';
import { UploadCloud, CheckCircle, AlertTriangle, FileText, ShieldCheck, MailWarning } from 'lucide-react';

const BASE_URL = window.location.hostname === "localhost" 
  ? "http://localhost:8000" 
  : "https://ken2707-fake-review-analyzer.hf.space";

const MODEL_METRICS = {
  fake_review: {
    lstm: { name: 'LSTM + Word2Vec', acc: '93.71%', prec: '94.55%', rec: '92.75%', f1: '93.64%' },
    lr: { name: 'Logistic Regression', acc: '92.73%', prec: '93.31%', rec: '92.06%', f1: '92.68%' },
    rf: { name: 'Random Forest', acc: '89.42%', prec: '92.01%', rec: '86.32%', f1: '89.08%' },
    xgb: { name: 'XGBoost', acc: '88.85%', prec: '91.17%', rec: '86.03%', f1: '88.52%' }
  },
  spam: {
    lstm: { name: 'LSTM + Word2Vec', acc: '98.30%', prec: '96.43%', rec: '90.60%', f1: '93.43%' },
    rf: { name: 'Random Forest', acc: '97.13%', prec: '98.35%', rec: '79.87%', f1: '88.15%' },
    xgb: { name: 'XGBoost', acc: '97.49%', prec: '93.53%', rec: '87.25%', f1: '90.28%' },
    lr: { name: 'Logistic Regression', acc: '96.68%', prec: '97.46%', rec: '77.18%', f1: '86.14%' }
  }
};

function App() {
  const [activeTab, setActiveTab] = useState('manual'); 
  const [inputText, setInputText] = useState('');
  const [file, setFile] = useState(null);
  
  const [modelFake, setModelFake] = useState('lstm'); 
  const [modelSpam, setModelSpam] = useState('lstm'); 
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null); 
  const [bulkResults, setBulkResults] = useState([]);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setResult(null);
    setBulkResults([]);
  };

  const handleTextAnalyze = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setResult(null);
    
    try {
      const [resFake, resSpam] = await Promise.all([
        axios.post(`${BASE_URL}/api/analyze/text`, { text: inputText, mode: 'fake_review', model_type: modelFake }),
        axios.post(`${BASE_URL}/api/analyze/text`, { text: inputText, mode: 'spam', model_type: modelSpam })
      ]);
      
      setResult({
        fake: resFake.data.result,
        spam: resSpam.data.result
      });
    } catch (err) {
      alert("Error: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleFileAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setBulkResults([]);
    
    try {
      const [resFake, resSpam] = await Promise.all([
        axios.post(`${BASE_URL}/api/analyze/file`, (() => { const fd = new FormData(); fd.append("file", file); fd.append("mode", "fake_review"); fd.append("model_type", modelFake); return fd; })()),
        axios.post(`${BASE_URL}/api/analyze/file`, (() => { const fd = new FormData(); fd.append("file", file); fd.append("mode", "spam"); fd.append("model_type", modelSpam); return fd; })())
      ]);

      const combined = resFake.data.results.map((item, idx) => ({
        text: item.text,
        fake: item,
        spam: resSpam.data.results[idx]
      }));

      setBulkResults(combined);
    } catch (err) {
      alert("Error processing file: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const renderConfidenceBar = (confidence, isFlagged) => {
    const barColor = isFlagged ? 'bg-red-500' : 'bg-emerald-500';
    return (
      <div className="mt-3">
        <div className="flex justify-between text-sm mb-1 font-mono text-slate-400">
          <span>Confidence Score</span>
          <span className="font-bold text-slate-200">{confidence.toFixed(2)}%</span>
        </div>
        <div className="w-full bg-slate-800 rounded-full h-3.5 overflow-hidden border border-slate-700">
          <div className={`${barColor} h-3.5 rounded-full transition-all duration-1000`} style={{ width: `${confidence}%` }}></div>
        </div>
      </div>
    );
  };

  const renderMetrics = (metrics) => (
    <div className="grid grid-cols-4 gap-2 mt-2 text-center">
      <div className="bg-slate-900 p-2 rounded-lg border border-slate-700">
        <p className="text-[10px] text-slate-400 uppercase">Accuracy</p>
        <p className="text-xs font-bold text-teal-400">{metrics.acc}</p>
      </div>
      <div className="bg-slate-900 p-2 rounded-lg border border-slate-700">
        <p className="text-[10px] text-slate-400 uppercase">Precision</p>
        <p className="text-xs font-bold text-blue-400">{metrics.prec}</p>
      </div>
      <div className="bg-slate-900 p-2 rounded-lg border border-slate-700">
        <p className="text-[10px] text-slate-400 uppercase">Recall</p>
        <p className="text-xs font-bold text-purple-400">{metrics.rec}</p>
      </div>
      <div className="bg-slate-900 p-2 rounded-lg border border-slate-700">
        <p className="text-[10px] text-slate-400 uppercase">F1-Score</p>
        <p className="text-xs font-bold text-amber-400">{metrics.f1}</p>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen pb-16 pt-8 bg-slate-950 font-sans selection:bg-teal-500 selection:text-white">
      
      {/* HEADER */}
      <div className="max-w-4xl mx-auto text-center mb-8 px-4">
        <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-blue-500 tracking-tight mb-2">
          AI Text Sentinel
        </h1>
        <p className="text-slate-400 text-sm md:text-base mb-6">
          NLP Analyzer for Fake Review & Spam Detection
        </p>
        
        {/* TAB NAVIGATION */}
        <div className="inline-flex bg-slate-800 p-1 rounded-xl shadow-lg border border-slate-700">
          <button 
            onClick={() => handleTabChange('manual')}
            className={`px-8 py-2 rounded-lg text-sm font-bold transition ${activeTab === 'manual' ? 'bg-slate-700 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
          >
            Manual Input
          </button>
          <button 
            onClick={() => handleTabChange('file')}
            className={`px-8 py-2 rounded-lg text-sm font-bold transition ${activeTab === 'file' ? 'bg-slate-700 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
          >
            Choose File
          </button>
        </div>
      </div>

      <div className="w-[90%] md:w-[80%] mx-auto">
        
        {/* TAB: INPUT MANUAL */}
        {activeTab === 'manual' && (
          <div className="animate-fade-in">
            {/* INPUT BOX */}
            <div className="bg-slate-800 p-5 md:p-6 rounded-2xl border border-slate-700 shadow-2xl">
              <label className="block text-lg font-bold text-slate-200 mb-2 flex items-center gap-2">
                <FileText className="text-teal-400" size={20}/> Your text:
              </label>
              <textarea 
                rows="4"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Paste the review, message, or email here..."
                className="w-full bg-slate-900 border border-slate-700 rounded-xl p-4 text-slate-200 focus:outline-none focus:border-teal-500 text-lg shadow-inner"
              ></textarea>

              {/* MODEL CONFIGURATION  */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-5 border-t border-slate-700 pt-5">
                {/* Fake Review Config */}
                <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                  <label className="block text-sm font-bold text-slate-400 mb-1.5 flex items-center gap-2">
                    <ShieldCheck size={16} className="text-blue-400"/> Fake Review Engine
                  </label>
                  <select 
                    value={modelFake}
                    onChange={(e) => setModelFake(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-sm text-slate-200 focus:border-blue-500 outline-none"
                  >
                    <option value="lstm">🧠 Deep Learning: LSTM</option>
                    <option value="lr">📈 Logistic Regression</option>
                    <option value="rf">🌲 Random Forest</option>
                    <option value="xgb">🚀 XGBoost</option>
                  </select>
                  {renderMetrics(MODEL_METRICS.fake_review[modelFake])}
                </div>

                {/* Spam Config */}
                <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                  <label className="block text-sm font-bold text-slate-400 mb-1.5 flex items-center gap-2">
                    <MailWarning size={16} className="text-amber-400"/> Spam Detection Engine
                  </label>
                  <select 
                    value={modelSpam}
                    onChange={(e) => setModelSpam(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-sm text-slate-200 focus:border-amber-500 outline-none"
                  >
                    <option value="lstm">🧠 Deep Learning: LSTM</option>
                    <option value="lr">📈 Logistic Regression</option>
                    <option value="rf">🌲 Random Forest</option>
                    <option value="xgb">🚀 XGBoost</option>
                  </select>
                  {renderMetrics(MODEL_METRICS.spam[modelSpam])}
                </div>
              </div>

              {/* ACTION BUTTON */}
              <button 
                onClick={handleTextAnalyze}
                disabled={loading || !inputText}
                className="w-full mt-6 bg-gradient-to-r from-teal-600 to-blue-600 hover:from-teal-500 hover:to-blue-500 disabled:opacity-50 text-white font-bold py-3 rounded-xl text-lg shadow-lg shadow-blue-900/50 transition-all active:scale-[0.99]"
              >
                {loading ? 'Running AI Engines...' : 'Analyze Text'}
              </button>
            </div>

            {/* RESULTS AREA */}
            {result && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mt-5 animate-fade-in">
                {/* Fake Review Result */}
                <div className="bg-slate-800 p-5 md:p-6 rounded-2xl border border-slate-700 shadow-xl">
                  <h3 className="font-bold text-slate-400 border-b border-slate-700 pb-2 mb-4 uppercase tracking-widest flex items-center gap-2">
                    <ShieldCheck size={18}/> Fake Review Report
                  </h3>
                  <div className={`p-4 rounded-xl border flex gap-3 items-center ${result.fake.is_flagged ? 'bg-red-950/30 border-red-900/50' : 'bg-emerald-950/30 border-emerald-900/50'}`}>
                    {result.fake.is_flagged ? <AlertTriangle className="text-red-500 flex-shrink-0" size={28}/> : <CheckCircle className="text-emerald-500 flex-shrink-0" size={28}/>}
                    <div>
                      <p className={`text-xl font-black ${result.fake.is_flagged ? 'text-red-400' : 'text-emerald-400'}`}>{result.fake.label}</p>
                    </div>
                  </div>
                  {renderConfidenceBar(result.fake.confidence, result.fake.is_flagged)}
                  <div className="mt-4">
                    <p className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-1.5">Reasoning:</p>
                    <p className="text-sm text-slate-300 italic bg-slate-900 p-3 rounded-lg border border-slate-800">"{result.fake.reason}"</p>
                  </div>
                </div>

                {/* Spam Result */}
                <div className="bg-slate-800 p-5 md:p-6 rounded-2xl border border-slate-700 shadow-xl">
                  <h3 className="font-bold text-slate-400 border-b border-slate-700 pb-2 mb-4 uppercase tracking-widest flex items-center gap-2">
                    <MailWarning size={18}/> Spam Report
                  </h3>
                  <div className={`p-4 rounded-xl border flex gap-3 items-center ${result.spam.is_flagged ? 'bg-red-950/30 border-red-900/50' : 'bg-emerald-950/30 border-emerald-900/50'}`}>
                    {result.spam.is_flagged ? <AlertTriangle className="text-red-500 flex-shrink-0" size={28}/> : <CheckCircle className="text-emerald-500 flex-shrink-0" size={28}/>}
                    <div>
                      <p className={`text-xl font-black ${result.spam.is_flagged ? 'text-red-400' : 'text-emerald-400'}`}>{result.spam.label}</p>
                    </div>
                  </div>
                  {renderConfidenceBar(result.spam.confidence, result.spam.is_flagged)}
                  <div className="mt-4">
                    <p className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-1.5">Reasoning:</p>
                    <p className="text-sm text-slate-300 italic bg-slate-900 p-3 rounded-lg border border-slate-800">"{result.spam.reason}"</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* CHOOSE FILE */}
        {activeTab === 'file' && (
          <div className="bg-slate-800 p-5 md:p-8 rounded-2xl border border-slate-700 shadow-2xl animate-fade-in">
             
             <div className="mb-6">
               <h2 className="text-2xl font-bold text-white mb-1.5 flex items-center gap-2"><UploadCloud/> Batch Analysis (.csv / .txt)</h2>
               <p className="text-slate-400 text-sm">Upload a file to run both Fake Review and Spam detection models simultaneously across all rows.</p>
             </div>

             {/* 1. FILE UPLOAD BOX */}
             <div className="border-2 border-dashed border-slate-600 rounded-xl p-6 text-center bg-slate-900/50 hover:bg-slate-900 transition mb-6">
                <input 
                  type="file" 
                  accept=".csv,.txt"
                  onChange={(e) => setFile(e.target.files[0])}
                  className="block w-full text-slate-400 file:mr-4 file:py-2.5 file:px-6 file:rounded-full file:border-0 file:text-sm file:font-bold file:bg-teal-600 file:text-white hover:file:bg-teal-500 cursor-pointer"
                />
             </div>
             
             {/* 2. MODEL Config */}
             <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-6 border-t border-slate-700 pt-6">
                <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700">
                  <label className="block text-sm font-bold text-slate-400 mb-2 flex items-center gap-2">
                    <ShieldCheck size={16} className="text-blue-400"/> Fake Review Engine
                  </label>
                  <select value={modelFake} onChange={(e) => setModelFake(e.target.value)} className="w-full bg-slate-800 border border-slate-700 outline-none rounded-lg p-2 text-sm text-slate-200">
                    <option value="lstm">🧠 Deep Learning: LSTM</option>
                    <option value="lr">📈 Logistic Regression</option>
                    <option value="rf">🌲 Random Forest</option>
                    <option value="xgb">🚀 XGBoost</option>
                  </select>
                  {renderMetrics(MODEL_METRICS.fake_review[modelFake])}
                </div>
                
                <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700">
                  <label className="block text-sm font-bold text-slate-400 mb-2 flex items-center gap-2">
                    <MailWarning size={16} className="text-amber-400"/> Spam Detection Engine
                  </label>
                  <select value={modelSpam} onChange={(e) => setModelSpam(e.target.value)} className="w-full bg-slate-800 border border-slate-700 outline-none rounded-lg p-2 text-sm text-slate-200">
                    <option value="lstm">🧠 Deep Learning: LSTM</option>
                    <option value="lr">📈 Logistic Regression</option>
                    <option value="rf">🌲 Random Forest</option>
                    <option value="xgb">🚀 XGBoost</option>
                  </select>
                  {renderMetrics(MODEL_METRICS.spam[modelSpam])}
                </div>
             </div>

             <button 
                onClick={handleFileAnalyze}
                disabled={loading || !file}
                className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-bold py-3 rounded-xl text-lg shadow-lg transition-all"
             >
                {loading ? 'Processing Document...' : 'Run Batch Analysis'}
             </button>

             {/* Bulk Results Table */}
             {bulkResults.length > 0 && (
                <div className="mt-8 border-t border-slate-700 pt-6 animate-fade-in">
                  <h3 className="text-xl font-bold text-white mb-3">Results ({bulkResults.length} items)</h3>
                  <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                    {bulkResults.map((item, idx) => (
                      <div key={idx} className="bg-slate-900 p-4 rounded-xl border border-slate-700 text-sm">
                        <p className="text-slate-300 italic mb-4 line-clamp-3">"{item.text}"</p>
                        <div className="grid grid-cols-2 gap-6 border-t border-slate-800 pt-3">
                          {/* Fake Review Bar */}
                          <div>
                            <span className="text-[10px] text-slate-500 uppercase block mb-1">Fake Review</span>
                            <div className="flex justify-between items-end mb-1">
                              <span className={`text-xs font-black uppercase tracking-wider ${item.fake.is_flagged ? 'text-red-400' : 'text-emerald-400'}`}>
                                {item.fake.label}
                              </span>
                              <span className="text-[10px] font-mono text-slate-400">{item.fake.confidence.toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden border border-slate-700">
                              <div className={`${item.fake.is_flagged ? 'bg-red-500' : 'bg-emerald-500'} h-1.5 rounded-full transition-all duration-1000`} style={{ width: `${item.fake.confidence}%` }}></div>
                            </div>
                          </div>

                          {/* Spam Detection Bar */}
                          <div>
                            <span className="text-[10px] text-slate-500 uppercase block mb-1">Spam Detection</span>
                            <div className="flex justify-between items-end mb-1">
                              <span className={`text-xs font-black uppercase tracking-wider ${item.spam.is_flagged ? 'text-red-400' : 'text-emerald-400'}`}>
                                {item.spam.label}
                              </span>
                              <span className="text-[10px] font-mono text-slate-400">{item.spam.confidence.toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden border border-slate-700">
                              <div className={`${item.spam.is_flagged ? 'bg-red-500' : 'bg-emerald-500'} h-1.5 rounded-full transition-all duration-1000`} style={{ width: `${item.spam.confidence}%` }}></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
             )}
         </div>
        )}
      </div>
    </div>
  );
}

export default App;