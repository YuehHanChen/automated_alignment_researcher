import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import IdeaList from './IdeaList';
import QueuePanel from './QueuePanel';
import Leaderboard from './Leaderboard';
import Forum from './Forum';

// Use empty string for relative paths (production), or localhost:8000 for development
const API_BASE = process.env.REACT_APP_API_URL !== undefined
  ? process.env.REACT_APP_API_URL
  : 'http://localhost:8000';

// Anthropic Light Theme - matching anthropic.com
const theme = {
  bgPrimary: '#FAF9F7',
  bgSecondary: '#FFFFFF',
  bgTertiary: '#F5F4F2',
  bgElevated: '#EEEEED',
  textPrimary: '#191918',
  textSecondary: '#5C5C5A',
  textTertiary: '#8C8C8A',
  borderSubtle: '#E8E7E5',
  borderDefault: '#D4D3D1',
  accentCoral: '#C4704F',
  accentCoralHover: '#B5613F',
  accentGreen: '#2E7D5A',
  accentRed: '#C54B4B',
  accentBlue: '#4B7CC5',
  accentAmber: '#B58B3D',
  accentPurple: '#7B5CB5',
};

function App() {
  const [ideas, setIdeas] = useState([]);
  const [queue, setQueue] = useState([]);
  const [leaderboard, setLeaderboard] = useState([]);
  const [stats, setStats] = useState({});
  const [config, setConfig] = useState({});
  const [loading, setLoading] = useState(true);
  const [configInitialized, setConfigInitialized] = useState(false); // Track if config has been loaded
  const [activeTab, setActiveTab] = useState('ideas');
  const [error, setError] = useState(null);
  
  // Selected configuration (dataset, weak_model, strong_model)
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [selectedWeakModel, setSelectedWeakModel] = useState(null);
  const [selectedStrongModel, setSelectedStrongModel] = useState(null);
  const [selectedExecutionMode, setSelectedExecutionMode] = useState('local');

  // Request counter to ignore stale responses (prevents race conditions)
  const requestIdRef = useRef(0);

  // Helper to fetch data with explicit params (avoids stale closure issues)
  const fetchDataWithParams = useCallback(async (dataset, weakModel, strongModel) => {
    const currentRequestId = ++requestIdRef.current;
    
    try {
      // Build query params for config filtering
      const params = new URLSearchParams();
      if (dataset) params.append('dataset', dataset);
      if (weakModel) params.append('weak_model', weakModel);
      if (strongModel) params.append('strong_model', strongModel);
      const queryString = params.toString() ? `?${params.toString()}` : '';
      
      console.log(`[fetchData] Request ${currentRequestId}: dataset=${dataset}, weak=${weakModel}, strong=${strongModel}`);
      
      // Fetch all data with config filter
      const [ideasRes, queueRes, leaderboardRes, statsRes] = await Promise.all([
        axios.get(`${API_BASE}/api/ideas${queryString}`),
        axios.get(`${API_BASE}/api/queue${queryString}`),
        axios.get(`${API_BASE}/api/leaderboard${queryString}`),
        axios.get(`${API_BASE}/api/stats${queryString}`)
      ]);

      // Ignore stale responses - only update if this is the latest request
      if (currentRequestId !== requestIdRef.current) {
        console.log(`[fetchData] Ignoring stale response (request ${currentRequestId}, current ${requestIdRef.current})`);
        return;
      }

      console.log(`[fetchData] Request ${currentRequestId} completed: queue=${queueRes.data.experiments?.length}, leaderboard=${leaderboardRes.data.experiments?.length}`);
      
      setIdeas(ideasRes.data.ideas || []);
      setQueue(queueRes.data.experiments || []);
      setLeaderboard(leaderboardRes.data.experiments || []);
      setStats(statsRes.data || {});
      setError(null);
    } catch (err) {
      // Ignore errors from stale requests
      if (currentRequestId !== requestIdRef.current) return;
      
      console.error('Error fetching data:', err);
      setError(err.message);
    } finally {
      // Only update loading state for the latest request
      if (currentRequestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }, []); // No dependencies - params are passed explicitly

  // Initialize config and fetch initial data on mount
  useEffect(() => {
    const initConfig = async () => {
      try {
        const configRes = await axios.get(`${API_BASE}/api/config`);
        const configData = configRes.data || {};
        
        // Get the config values we'll use
        const dataset = configData.dataset || null;
        const weakModel = configData.weak_model || null;
        const strongModel = configData.strong_model || null;
        
        console.log(`[initConfig] Loaded config: dataset=${dataset}, weak=${weakModel}, strong=${strongModel}`);
        
        // Set all state
        setConfig(configData);
        setSelectedDataset(dataset);
        setSelectedWeakModel(weakModel);
        setSelectedStrongModel(strongModel);

        setConfigInitialized(true);
        
        // Fetch initial data directly with config values (not relying on state)
        await fetchDataWithParams(dataset, weakModel, strongModel);
        
      } catch (err) {
        console.error('Error fetching config:', err);
        setError(err.message);
        setLoading(false);
      }
    };
    
    initConfig();
  }, [fetchDataWithParams]); // Only run once on mount

  // Fetch data when selections change (after initial load)
  useEffect(() => {
    // Skip if not initialized yet (initial fetch is handled by initConfig)
    if (!configInitialized) return;
    
    // Fetch with current selection values
    fetchDataWithParams(selectedDataset, selectedWeakModel, selectedStrongModel);
  }, [configInitialized, selectedDataset, selectedWeakModel, selectedStrongModel, fetchDataWithParams]);

  // Polling interval - only active after config is initialized
  useEffect(() => {
    if (!configInitialized) return;
    
    const interval = setInterval(() => {
      fetchDataWithParams(selectedDataset, selectedWeakModel, selectedStrongModel);
    }, 5000);
    return () => clearInterval(interval);
  }, [configInitialized, selectedDataset, selectedWeakModel, selectedStrongModel, fetchDataWithParams]);

  // Simple wrapper for handlers that fetches with current state values
  const fetchData = useCallback(async () => {
    if (!configInitialized) return;
    await fetchDataWithParams(selectedDataset, selectedWeakModel, selectedStrongModel);
  }, [configInitialized, selectedDataset, selectedWeakModel, selectedStrongModel, fetchDataWithParams]);

  const handleAddToQueue = async (idea) => {
    try {
      await axios.post(`${API_BASE}/api/queue/add`, {
        idea_name: idea.Name,
        idea_title: idea.Name,
        idea_description: idea.Description || idea.Name,
        dataset: selectedDataset,
        weak_model: selectedWeakModel,
        strong_model: selectedStrongModel,
        execution_mode: selectedExecutionMode,
      });
      await fetchData();
    } catch (err) {
      console.error('Error adding to queue:', err);
      alert(`Failed to add to queue: ${err.response?.data?.error || err.message}`);
    }
  };

  const handleRemoveFromQueue = async (experimentId) => {
    if (!window.confirm('Are you sure you want to remove this experiment from the queue?')) {
      return;
    }
    try {
      await axios.delete(`${API_BASE}/api/queue/remove/${experimentId}`);
      await fetchData();
    } catch (err) {
      console.error('Error removing from queue:', err);
      alert(`Failed to remove from queue: ${err.response?.data?.error || err.message}`);
    }
  };

  const handleRerunExperiment = async (experimentId) => {
    if (!window.confirm('Are you sure you want to rerun this experiment?')) {
      return;
    }
    try {
      await axios.post(`${API_BASE}/api/queue/rerun/${experimentId}`);
      await fetchData();
    } catch (err) {
      console.error('Error rerunning experiment:', err);
      alert(`Failed to rerun experiment: ${err.response?.data?.error || err.message}`);
    }
  };

  const handleKillExperiment = async (experimentId) => {
    if (!window.confirm('Are you sure you want to kill this running experiment?')) {
      return;
    }
    try {
      await axios.post(`${API_BASE}/api/queue/kill/${experimentId}`);
      await fetchData();
    } catch (err) {
      console.error('Error killing experiment:', err);
      alert(`Failed to kill experiment: ${err.response?.data?.error || err.message}`);
    }
  };

  // Handler for dataset change - updates both local state and backend config
  const handleDatasetChange = async (value) => {
    setSelectedDataset(value);
    try {
      await axios.post(`${API_BASE}/api/config`, { dataset: value });
      console.log(`[Config] Dataset updated to: ${value}`);
    } catch (err) {
      console.error('Error updating dataset config:', err);
    }
  };

  // Handler for weak model change - updates both local state and backend config
  const handleWeakModelChange = async (value) => {
    setSelectedWeakModel(value);
    try {
      await axios.post(`${API_BASE}/api/config`, { weak_model: value });
      console.log(`[Config] Weak model updated to: ${value}`);
    } catch (err) {
      console.error('Error updating weak_model config:', err);
    }
  };

  // Handler for strong model change - updates both local state and backend config
  const handleStrongModelChange = async (value) => {
    setSelectedStrongModel(value);
    try {
      await axios.post(`${API_BASE}/api/config`, { strong_model: value });
      console.log(`[Config] Strong model updated to: ${value}`);
    } catch (err) {
      console.error('Error updating strong_model config:', err);
    }
  };

  // Handler for max concurrent pods change
  const handleMaxConcurrentPodsChange = async (value) => {
    const numValue = parseInt(value, 10);
    if (isNaN(numValue) || numValue < 1 || numValue > 5) return;
    try {
      await axios.post(`${API_BASE}/api/config`, { max_concurrent_pods: numValue });
      setConfig(prev => ({ ...prev, max_concurrent_pods: numValue }));
      console.log(`[Config] Max concurrent pods updated to: ${numValue}`);
    } catch (err) {
      console.error('Error updating max_concurrent_pods:', err);
      alert(`Failed to update: ${err.response?.data?.error || err.message}`);
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh',
      background: theme.bgPrimary,
    }}>
      {/* Header */}
      <header style={{
        padding: '48px 48px 40px',
        borderBottom: `1px solid ${theme.borderSubtle}`,
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
        }}>
          <h1 style={{
            margin: 0,
            fontSize: '36px',
            fontWeight: '600',
            color: theme.textPrimary,
            letterSpacing: '-0.02em',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
          }}>
            {/* Flask/Science Icon */}
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke={theme.accentCoral} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 3h6v8l4 9H5l4-9V3z" />
              <path d="M9 3h6" />
              <path d="M10 12h4" />
            </svg>
            MoreWrong
          </h1>
          <p style={{
            margin: '16px 0 0',
            fontSize: '18px',
            color: theme.textSecondary,
            lineHeight: '1.6',
            maxWidth: '700px',
          }}>
            Learn from Mistakes
          </p>

          {/* Config Selectors */}
          {config.dataset && (
            <div style={{
              marginTop: '36px',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '24px',
            }}>
              {/* Dataset Selector */}
              <ConfigSelector
                label="Dataset"
                value={selectedDataset || config.dataset || ''}
                onChange={handleDatasetChange}
                options={config.available_datasets || [config.dataset]}
                theme={theme}
              />
              
              {/* Weak Model Selector */}
              <ConfigSelector
                label="Weak Model"
                value={selectedWeakModel || config.weak_model || ''}
                onChange={handleWeakModelChange}
                options={config.available_weak_models || []}
                theme={theme}
              />
              
              {/* Strong Model Selector */}
              <ConfigSelector
                label="Strong Model"
                value={selectedStrongModel || config.strong_model || ''}
                onChange={handleStrongModelChange}
                options={config.available_strong_models || [config.strong_model]}
                theme={theme}
              />

              {/* Execution Mode Selector */}
              <ConfigSelector
                label="Execution Mode"
                value={selectedExecutionMode}
                onChange={(val) => setSelectedExecutionMode(val)}
                options={['local', 'docker', 'runpod']}
                theme={theme}
              />

              {/* Max Jobs Selector */}
              <div>
                <div style={{
                  fontSize: '12px',
                  color: theme.textTertiary,
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                  marginBottom: '8px',
                  fontWeight: '500',
                }}>
                  Max Jobs
                </div>
                <select
                  value={config.max_concurrent_pods || 5}
                  onChange={(e) => handleMaxConcurrentPodsChange(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px 14px',
                    fontSize: '14px',
                    fontWeight: '500',
                    color: theme.textPrimary,
                    background: theme.bgSecondary,
                    border: `1px solid ${theme.borderDefault}`,
                    borderRadius: '8px',
                    cursor: 'pointer',
                  }}
                >
                  {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(n => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div style={{
          background: '#FEF2F2',
          borderBottom: `1px solid ${theme.accentRed}40`,
          padding: '14px 48px',
        }}>
          <div style={{
            maxWidth: '1200px',
            margin: '0 auto',
            color: theme.accentRed,
            fontSize: '14px',
          }}>
            <span style={{ marginRight: '8px' }}>⚠</span>
            {error}
          </div>
        </div>
      )}

      {/* Stats Bar */}
      <div style={{
        padding: '28px 48px',
        borderBottom: `1px solid ${theme.borderSubtle}`,
        background: theme.bgSecondary,
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          display: 'flex',
          gap: '56px',
          alignItems: 'center',
        }}>
          <StatBox label="Total" value={stats.total || 0} color={theme.textPrimary} theme={theme} />
          <StatBox label="Queued" value={stats.queued || 0} color={theme.accentAmber} theme={theme} />
          <StatBox label="Running" value={stats.running || 0} color={theme.accentBlue} theme={theme} />
          <StatBox label="Completed" value={stats.completed || 0} color={theme.accentGreen} theme={theme} />
          <StatBox label="Failed" value={stats.failed || 0} color={theme.accentRed} theme={theme} />
        </div>
      </div>

      {/* Tab Navigation */}
      <div style={{
        padding: '0 48px',
        borderBottom: `1px solid ${theme.borderSubtle}`,
        background: theme.bgSecondary,
      }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          display: 'flex',
          gap: '8px',
        }}>
          {[
            { id: 'ideas', label: 'Ideas', count: ideas.length },
            { id: 'queue', label: 'Queue', count: queue.length },
            { id: 'leaderboard', label: 'Leaderboard', count: leaderboard.length },
            { id: 'forum', label: 'Forum', count: null }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '18px 24px',
                background: 'transparent',
                border: 'none',
                borderBottom: activeTab === tab.id 
                  ? `2px solid ${theme.textPrimary}` 
                  : '2px solid transparent',
                color: activeTab === tab.id ? theme.textPrimary : theme.textTertiary,
                fontWeight: '500',
                fontSize: '15px',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                marginBottom: '-1px',
              }}
            >
              {tab.label}
              {tab.count !== null && (
                <span style={{
                  marginLeft: '10px',
                  padding: '3px 10px',
                  background: activeTab === tab.id ? theme.bgTertiary : theme.bgElevated,
                  borderRadius: '12px',
                  fontSize: '13px',
                  fontWeight: '600',
                  color: activeTab === tab.id ? theme.textPrimary : theme.textTertiary,
                }}>
                  {tab.count}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <main style={{
        padding: '48px',
        maxWidth: (activeTab === 'leaderboard' || activeTab === 'forum') ? '1400px' : '1200px',
        margin: '0 auto',
      }}>
        {activeTab === 'ideas' && (
          <IdeaList
            ideas={ideas}
            onAddToQueue={handleAddToQueue}
            loading={loading}
            onIdeaAdded={async (response) => {
              await fetchData();
              if (response.experiment) {
                alert(`Idea "${response.idea.Name}" created and added to queue!`);
              } else {
                alert(`Idea "${response.idea.Name}" created successfully!`);
              }
            }}
            theme={theme}
            dataset={selectedDataset}
            weakModel={selectedWeakModel}
            strongModel={selectedStrongModel}
          />
        )}

        {activeTab === 'queue' && (
          <QueuePanel
            experiments={queue}
            onRemove={handleRemoveFromQueue}
            onRerun={handleRerunExperiment}
            onKill={handleKillExperiment}
            loading={loading}
            theme={theme}
          />
        )}

        {activeTab === 'leaderboard' && (
          <Leaderboard
            experiments={leaderboard}
            loading={loading}
            theme={theme}
          />
        )}

        {activeTab === 'forum' && (
          <Forum theme={theme} />
        )}
      </main>

      {/* Footer */}
      <footer style={{
        padding: '24px 48px',
        textAlign: 'center',
        color: theme.textTertiary,
        fontSize: '13px',
        borderTop: `1px solid ${theme.borderSubtle}`,
      }}>
        W2S Research v1.0
      </footer>

    </div>
  );
}

// Config Selector Component (dropdown) - styled custom select
const ConfigSelector = ({ label, value, onChange, options, theme }) => (
  <div>
    <div style={{
      fontSize: '12px',
      color: theme.textTertiary,
      textTransform: 'uppercase',
      letterSpacing: '0.08em',
      marginBottom: '8px',
      fontWeight: '500',
    }}>
      {label}
    </div>
    <div style={{ position: 'relative', display: 'inline-block', width: '100%' }}>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: '100%',
          padding: '10px 36px 10px 14px',
          fontSize: '14px',
          fontWeight: '500',
          color: theme.textPrimary,
          background: theme.bgSecondary,
          border: `1px solid ${theme.borderDefault}`,
          borderRadius: '8px',
          cursor: 'pointer',
          appearance: 'none',
          WebkitAppearance: 'none',
          MozAppearance: 'none',
        }}
      >
        {options.map(opt => (
          <option key={opt} value={opt}>
            {opt.includes('/') ? opt.split('/').pop() : opt}
          </option>
        ))}
      </select>
      {/* Custom dropdown arrow */}
      <svg 
        width="16" 
        height="16" 
        viewBox="0 0 24 24" 
        fill="none" 
        stroke={theme.textTertiary}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{
          position: 'absolute',
          right: '12px',
          top: '50%',
          transform: 'translateY(-50%)',
          pointerEvents: 'none',
        }}
      >
        <polyline points="6 9 12 15 18 9" />
      </svg>
    </div>
  </div>
);

// Stat Box Component
const StatBox = ({ label, value, color, theme }) => (
  <div>
    <div style={{ 
      fontSize: '36px', 
      fontWeight: '600', 
      color,
      letterSpacing: '-0.02em',
      fontFamily: "'JetBrains Mono', monospace",
    }}>
      {value}
    </div>
    <div style={{ 
      fontSize: '12px', 
      color: theme.textTertiary, 
      textTransform: 'uppercase',
      letterSpacing: '0.08em',
      marginTop: '4px',
      fontWeight: '500',
    }}>
      {label}
    </div>
  </div>
);

export default App;
