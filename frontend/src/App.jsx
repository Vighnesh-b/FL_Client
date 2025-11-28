import React, { useState, useEffect } from 'react';
import { Activity, CheckCircle, XCircle, RefreshCw, Send, Download, Server, Terminal, AlertCircle } from 'lucide-react';

const ClientDashboard = () => {
  const [clientId, setClientId] = useState('client_1');
  const [currentRound, setCurrentRound] = useState(1);
  const [status, setStatus] = useState('idle'); // idle, training, uploading, complete
  const [logs, setLogs] = useState([]);
  const [trainingMetrics, setTrainingMetrics] = useState({
    trainLoss: 0,
    valLoss: 0,
    valDice: 0,
    epoch: 0
  });
  const [serverUrl, setServerUrl] = useState('http://127.0.0.1:8000');
  const [localBackendUrl] = useState('http://127.0.0.1:5000');
  const [connectionStatus, setConnectionStatus] = useState({
    localBackend: false,
    server: false
  });

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { message, type, timestamp }]);
  };

  // Check connection to local backend and server
  const checkConnections = async () => {
    try {
      const response = await fetch(`${localBackendUrl}/health`, { method: 'GET' });
      setConnectionStatus(prev => ({ ...prev, localBackend: response.ok }));
    } catch (error) {
      setConnectionStatus(prev => ({ ...prev, localBackend: false }));
    }

    try {
      const response = await fetch(`${serverUrl}/api/get-global-model`, { method: 'HEAD' });
      setConnectionStatus(prev => ({ ...prev, server: response.ok || response.status === 404 }));
    } catch (error) {
      setConnectionStatus(prev => ({ ...prev, server: false }));
    }
  };

  useEffect(() => {
    checkConnections();
    const interval = setInterval(checkConnections, 10000);
    return () => clearInterval(interval);
  }, [serverUrl]);

  // Simulate downloading global model
  const downloadGlobalModel = async () => {
    addLog(`Downloading global model from ${serverUrl}...`, 'info');
    setStatus('downloading');

    try {
      const response = await fetch(`${serverUrl}/api/get-global-model`);
      
      if (!response.ok) {
        throw new Error('Failed to download global model');
      }

      // Simulate download progress
      const reader = response.body.getReader();
      const contentLength = +response.headers.get('Content-Length');
      let receivedLength = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        receivedLength += value.length;
        const progress = Math.round((receivedLength / contentLength) * 100);
        addLog(`Download progress: ${progress}%`, 'info');
      }

      addLog('Global model downloaded successfully!', 'success');
      return true;
    } catch (error) {
      addLog(`Download failed: ${error.message}`, 'error');
      setStatus('idle');
      return false;
    }
  };

  // Simulate local training
  const trainLocalModel = async () => {
    addLog(`Starting local training for Round ${currentRound}...`, 'info');
    setStatus('training');

    const epochs = 3;
    for (let epoch = 1; epoch <= epochs; epoch++) {
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate training time

      const trainLoss = (Math.random() * 0.5 + 0.3).toFixed(4);
      const valLoss = (Math.random() * 0.4 + 0.35).toFixed(4);
      const valDice = (Math.random() * 0.2 + 0.75).toFixed(4);

      setTrainingMetrics({
        trainLoss: parseFloat(trainLoss),
        valLoss: parseFloat(valLoss),
        valDice: parseFloat(valDice),
        epoch
      });

      addLog(`Epoch ${epoch}/${epochs} - Train Loss: ${trainLoss}, Val Loss: ${valLoss}, Val Dice: ${valDice}`, 'success');
    }

    addLog('Local training completed!', 'success');
    return true;
  };

  // Upload local model to server
  const uploadLocalModel = async () => {
    addLog('Uploading local model weights...', 'info');
    setStatus('uploading');

    try {
      // Create a dummy file to simulate model weights
      const modelData = new Blob(['dummy model weights'], { type: 'application/octet-stream' });
      const formData = new FormData();
      formData.append('file', modelData, `${clientId}_round${currentRound}.pth`);
      formData.append('client_id', clientId);
      formData.append('dataset_size', '56'); // From config
      formData.append('cur_round', currentRound.toString());
      formData.append('federated_server_url', serverUrl);

      const response = await fetch(`${localBackendUrl}/api/send-local-model`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const result = await response.json();
      addLog('Local model uploaded successfully!', 'success');
      return true;
    } catch (error) {
      addLog(`Upload failed: ${error.message}`, 'error');
      setStatus('idle');
      return false;
    }
  };

  // Run complete federated learning round
  const runRound = async () => {
    if (status !== 'idle') {
      addLog('Round already in progress!', 'error');
      return;
    }

    addLog(`===== Starting Round ${currentRound} =====`, 'info');

    // Step 1: Download global model
    const downloaded = await downloadGlobalModel();
    if (!downloaded) return;

    // Step 2: Train locally
    const trained = await trainLocalModel();
    if (!trained) return;

    // Step 3: Upload weights
    const uploaded = await uploadLocalModel();
    if (!uploaded) return;

    // Complete
    addLog(`===== Round ${currentRound} Completed =====`, 'success');
    setStatus('idle');
    setCurrentRound(prev => prev + 1);
  };

  const resetClient = () => {
    setCurrentRound(1);
    setStatus('idle');
    setLogs([]);
    setTrainingMetrics({ trainLoss: 0, valLoss: 0, valDice: 0, epoch: 0 });
    addLog('Client reset successfully', 'info');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2 flex items-center gap-3">
                <Activity className="text-purple-400" />
                Federated Learning Client
              </h1>
              <p className="text-purple-300">Brain Tumor Segmentation - UNETR Model</p>
            </div>
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
              <div className="text-sm text-slate-400 mb-1">Client ID</div>
              <input
                type="text"
                value={clientId}
                onChange={(e) => setClientId(e.target.value)}
                className="px-4 py-2 bg-white/5 border border-white/20 rounded-lg text-white font-mono focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none"
              />
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* Connection Status Card */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 hover:bg-white/15 transition-all">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-blue-500/20 rounded-lg">
                <Server className="w-6 h-6 text-blue-400" />
              </div>
              <div className="flex gap-2">
                {connectionStatus.localBackend ? (
                  <CheckCircle className="text-green-400" size={20} />
                ) : (
                  <XCircle className="text-red-400" size={20} />
                )}
                {connectionStatus.server ? (
                  <CheckCircle className="text-green-400" size={20} />
                ) : (
                  <XCircle className="text-red-400" size={20} />
                )}
              </div>
            </div>
            <h3 className="text-2xl font-bold text-white mb-1">
              {connectionStatus.localBackend && connectionStatus.server ? 'Connected' : 'Disconnected'}
            </h3>
            <p className="text-slate-300 text-sm">Backend & Server</p>
          </div>

          {/* Current Round Card */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 hover:bg-white/15 transition-all">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-purple-500/20 rounded-lg">
                <RefreshCw className="w-6 h-6 text-purple-400" />
              </div>
              <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                status === 'idle' ? 'bg-gray-500/20 text-gray-300' :
                status === 'training' ? 'bg-amber-500/20 text-amber-300' :
                status === 'uploading' ? 'bg-blue-500/20 text-blue-300' :
                'bg-green-500/20 text-green-300'
              }`}>
                {status.toUpperCase()}
              </span>
            </div>
            <h3 className="text-3xl font-bold text-white mb-1">{currentRound}</h3>
            <p className="text-slate-300 text-sm">Current Round</p>
          </div>

          {/* Training Progress Card */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 hover:bg-white/15 transition-all">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-green-500/20 rounded-lg">
                <Activity className="w-6 h-6 text-green-400" />
              </div>
              <span className="text-green-300 text-sm font-medium">Epoch</span>
            </div>
            <h3 className="text-3xl font-bold text-white mb-1">{trainingMetrics.epoch}/3</h3>
            <p className="text-slate-300 text-sm">Training Progress</p>
          </div>

          {/* Dice Score Card */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 hover:bg-white/15 transition-all">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-pink-500/20 rounded-lg">
                <CheckCircle className="w-6 h-6 text-pink-400" />
              </div>
              <span className="text-pink-300 text-sm font-medium">Score</span>
            </div>
            <h3 className="text-3xl font-bold text-white mb-1">{trainingMetrics.valDice.toFixed(4)}</h3>
            <p className="text-slate-300 text-sm">Validation Dice</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Control Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 mb-6">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <Terminal className="w-5 h-5" />
                Control Panel
              </h2>

              <button
                onClick={runRound}
                disabled={status !== 'idle'}
                className={`w-full py-3 px-4 rounded-lg font-semibold transition-all flex items-center justify-center gap-2 mb-4 ${
                  status !== 'idle'
                    ? 'bg-gray-500/50 text-gray-300 cursor-not-allowed'
                    : 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white shadow-lg hover:shadow-xl'
                }`}
              >
                {status !== 'idle' ? (
                  <>
                    <RefreshCw className="w-5 h-5 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    Start Round {currentRound}
                  </>
                )}
              </button>

              <button
                onClick={resetClient}
                disabled={status !== 'idle'}
                className={`w-full py-3 px-4 rounded-lg font-semibold transition-all flex items-center justify-center gap-2 ${
                  status !== 'idle'
                    ? 'bg-gray-500/50 text-gray-300 cursor-not-allowed'
                    : 'bg-red-500/20 hover:bg-red-500/30 text-red-300 border border-red-500/30'
                }`}
              >
                <RefreshCw className="w-5 h-5" />
                Reset Client
              </button>

              {!connectionStatus.localBackend || !connectionStatus.server ? (
                <div className="bg-amber-500/20 border border-amber-500/30 rounded-lg p-3 mt-4">
                  <p className="text-amber-200 text-sm flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    Connection issues detected
                  </p>
                </div>
              ) : null}
            </div>

            {/* Training Metrics */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-bold text-white mb-4">Training Metrics</h2>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400 text-sm">Train Loss:</span>
                  <span className="text-white font-semibold">{trainingMetrics.trainLoss.toFixed(4)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400 text-sm">Val Loss:</span>
                  <span className="text-white font-semibold">{trainingMetrics.valLoss.toFixed(4)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400 text-sm">Val Dice:</span>
                  <span className="text-green-300 font-semibold">{trainingMetrics.valDice.toFixed(4)}</span>
                </div>
                <div className="flex justify-between items-center pt-3 border-t border-white/10">
                  <span className="text-slate-400 text-sm">Epoch:</span>
                  <span className="text-purple-300 font-semibold">{trainingMetrics.epoch}/3</span>
                </div>
              </div>

              <div className="mt-6 space-y-3">
                <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wide">Configuration</h3>
                <div className="space-y-2">
                  <div>
                    <span className="text-slate-400 text-xs">Local Backend:</span>
                    <div className="text-white font-mono text-xs mt-1 bg-white/5 px-2 py-1 rounded">{localBackendUrl}</div>
                  </div>
                  <div>
                    <span className="text-slate-400 text-xs">Server URL:</span>
                    <input
                      type="text"
                      value={serverUrl}
                      onChange={(e) => setServerUrl(e.target.value)}
                      className="w-full mt-1 px-2 py-1 bg-white/5 border border-white/20 rounded text-white font-mono text-xs focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Activity Log */}
          <div className="lg:col-span-2">
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Terminal className="w-5 h-5" />
                  Activity Log
                </h2>
                <button
                  onClick={() => setLogs([])}
                  className="text-sm text-slate-400 hover:text-white transition-colors"
                >
                  Clear
                </button>
              </div>

              <div className="bg-slate-950 rounded-lg p-4 h-[600px] overflow-y-auto custom-scrollbar font-mono text-sm">
                {logs.length === 0 ? (
                  <div className="text-center py-12">
                    <Terminal className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                    <p className="text-slate-500">No activity logs yet</p>
                    <p className="text-slate-600 text-xs mt-1">Start a round to begin training</p>
                  </div>
                ) : (
                  logs.map((log, idx) => (
                    <div key={idx} className="mb-2">
                      <span className="text-slate-500">[{log.timestamp}]</span>{' '}
                      <span className={
                        log.type === 'error' ? 'text-red-400' :
                        log.type === 'success' ? 'text-green-400' :
                        'text-blue-400'
                      }>
                        {log.message}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 8px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.05); border-radius: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.2); border-radius: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.3); }
      `}</style>
    </div>
  );
};

export default ClientDashboard;