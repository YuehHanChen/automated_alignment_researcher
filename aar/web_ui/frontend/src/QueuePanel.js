import React, { useState, useEffect, useMemo, useCallback } from 'react';
import axios from 'axios';
import StatusBadge from './StatusBadge';

const API_BASE = process.env.REACT_APP_API_URL !== undefined
  ? process.env.REACT_APP_API_URL
  : 'http://localhost:8000';

const defaultTheme = {
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
  accentGreen: '#2E7D5A',
  accentRed: '#C54B4B',
  accentBlue: '#4B7CC5',
  accentAmber: '#B58B3D',
};

const QueuePanel = ({ experiments, onRemove, onRerun, onKill, loading, theme = defaultTheme }) => {
  const [expandedId, setExpandedId] = useState(null);
  const [expandedSection, setExpandedSection] = useState({}); // { expId: 'console' | 's3log' | 's3results' | 'usage' }
  const [s3Data, setS3Data] = useState({}); // { expId: { log: string, results: object, loading: boolean } }
  const [usageStats, setUsageStats] = useState({}); // { expId: { stats: object, loading: boolean, error: string } }
  const [statusFilter, setStatusFilter] = useState('all'); // 'all', 'queued', 'running', 'completed', 'failed'

  // Helper: check if experiment is effectively failed (status=failed OR completed with no PGR)
  const isEffectivelyFailed = (exp) => {
    return exp.status === 'failed' || (exp.status === 'completed' && (exp.pgr === null || exp.pgr === undefined));
  };

  // Helper: check if experiment is truly completed (status=completed AND has valid PGR)
  const isTrulyCompleted = (exp) => {
    return exp.status === 'completed' && exp.pgr !== null && exp.pgr !== undefined;
  };

  // Sort experiments: newest first (by queue_time descending), then apply status filter
  // Note: Completed with null PGR is treated as failed (experiment ran but produced no results)
  const filteredExperiments = useMemo(() => {
    // Sort by queue_time descending (newest first, oldest at bottom)
    const sorted = [...experiments].sort((a, b) => {
      const timeA = a.queue_time ? new Date(a.queue_time).getTime() : 0;
      const timeB = b.queue_time ? new Date(b.queue_time).getTime() : 0;
      return timeB - timeA; // Descending order (newest first)
    });
    
    // Apply status filter
    if (statusFilter === 'all') return sorted;
    if (statusFilter === 'queued') return sorted.filter(exp => exp.status === 'queued');
    if (statusFilter === 'running') return sorted.filter(exp => exp.status === 'running');
    // Completed = completed AND has valid PGR
    if (statusFilter === 'completed') return sorted.filter(exp => isTrulyCompleted(exp));
    // Failed = failed OR (completed but no PGR)
    if (statusFilter === 'failed') return sorted.filter(exp => isEffectivelyFailed(exp));
    return sorted;
  }, [experiments, statusFilter]);

  // Count experiments by status
  // Note: Completed with null PGR is counted as failed
  const statusCounts = useMemo(() => {
    return {
      all: experiments.length,
      queued: experiments.filter(e => e.status === 'queued').length,
      running: experiments.filter(e => e.status === 'running').length,
      // Truly completed = completed AND has valid PGR
      completed: experiments.filter(e => isTrulyCompleted(e)).length,
      // Failed = failed OR (completed but no PGR)
      failed: experiments.filter(e => isEffectivelyFailed(e)).length,
    };
  }, [experiments]);

  const formatDuration = (seconds) => {
    if (!seconds) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  // Format datetime for display in Pacific time
  const formatDateTime = (isoString) => {
    if (!isoString) return null;
    // Treat timestamp as UTC by adding 'Z' if not present
    const utcString = isoString.endsWith('Z') ? isoString : isoString + 'Z';
    const date = new Date(utcString);
    const formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/Los_Angeles',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
    return formatter.format(date);
  };

  const toggleExpand = (id, section = 'console') => {
    const currentSection = expandedSection[id];
    if (currentSection === section) {
      // Collapse if clicking same section
      setExpandedSection({ ...expandedSection, [id]: null });
    } else {
      // Expand this section
      setExpandedSection({ ...expandedSection, [id]: section });

      // Fetch S3 data if needed
      if ((section === 's3log' || section === 's3results') && !s3Data[id]) {
        fetchS3Data(id);
      }

      // Fetch usage stats if needed
      if (section === 'usage' && !usageStats[id]) {
        fetchUsageStats(id);
      }
    }
  };

  const fetchS3Data = async (expId) => {
    const exp = experiments.find(e => e.id === expId);
    if (!exp || !exp.s3_prefix) return;

    setS3Data(prev => ({ ...prev, [expId]: { loading: true } }));

    try {
      const response = await axios.get(`${API_BASE}/api/experiment/${expId}/s3-data`);
      setS3Data(prev => ({
        ...prev,
        [expId]: {
          loading: false,
          log: response.data.worker_log,
          results: response.data.results,
          error: null
        }
      }));
    } catch (err) {
      console.error('Error fetching S3 data:', err);
      setS3Data(prev => ({
        ...prev,
        [expId]: {
          loading: false,
          error: err.response?.data?.error || err.message
        }
      }));
    }
  };

  const fetchUsageStats = useCallback(async (expId, isRefresh = false) => {
    // Only show loading spinner on initial fetch, not refreshes
    if (!isRefresh) {
      setUsageStats(prev => ({ ...prev, [expId]: { loading: true } }));
    }

    try {
      const response = await axios.get(`${API_BASE}/api/experiment/${expId}/usage-stats`);
      setUsageStats(prev => ({
        ...prev,
        [expId]: {
          loading: false,
          stats: response.data.usage_stats,
          error: null
        }
      }));
    } catch (err) {
      console.error('Error fetching usage stats:', err);
      setUsageStats(prev => ({
        ...prev,
        [expId]: {
          loading: false,
          stats: null,
          error: err.response?.data?.error || err.message
        }
      }));
    }
  }, []);

  // Auto-refresh usage stats every 30s for any expanded usage section
  useEffect(() => {
    const expandedUsageIds = Object.entries(expandedSection)
      .filter(([, section]) => section === 'usage')
      .map(([id]) => id);

    if (expandedUsageIds.length === 0) return;

    const interval = setInterval(() => {
      expandedUsageIds.forEach(id => fetchUsageStats(id, true));
    }, 30000);
    return () => clearInterval(interval);
  }, [expandedSection, fetchUsageStats]);

  if (loading) {
    return (
      <div style={{ 
        padding: '80px 20px', 
        textAlign: 'center',
        color: theme.textSecondary,
      }}>
        <div style={{ fontSize: '15px' }}>Loading queue...</div>
      </div>
    );
  }

  return (
    <div style={{
      background: theme.bgSecondary,
      borderRadius: '12px',
      border: `1px solid ${theme.borderSubtle}`,
      overflow: 'hidden'
    }}>
      <div style={{
        padding: '24px 28px',
        borderBottom: `1px solid ${theme.borderSubtle}`,
      }}>
        <h2 style={{ 
          margin: 0, 
          fontSize: '20px', 
          fontWeight: '600',
          color: theme.textPrimary,
          marginBottom: '20px',
        }}>
          Experiment Queue
        </h2>
        
        {/* Status Filter */}
        <div style={{
          display: 'flex',
          gap: '8px',
          flexWrap: 'wrap',
        }}>
          {[
            { key: 'all', label: 'All', color: theme.textPrimary },
            { key: 'queued', label: 'Queued', color: theme.accentAmber },
            { key: 'running', label: 'Running', color: theme.accentBlue },
            { key: 'completed', label: 'Completed', color: theme.accentGreen },
            { key: 'failed', label: 'Failed', color: theme.accentRed },
          ].map(filter => (
            <button
              key={filter.key}
              onClick={() => setStatusFilter(filter.key)}
              style={{
                padding: '6px 14px',
                background: statusFilter === filter.key ? `${filter.color}15` : theme.bgTertiary,
                color: statusFilter === filter.key ? filter.color : theme.textTertiary,
                border: statusFilter === filter.key ? `1px solid ${filter.color}40` : `1px solid ${theme.borderSubtle}`,
                borderRadius: '20px',
                fontSize: '13px',
                fontWeight: '500',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              {filter.label}
              <span style={{
                background: statusFilter === filter.key ? `${filter.color}25` : theme.bgElevated,
                padding: '2px 8px',
                borderRadius: '10px',
                fontSize: '12px',
                fontWeight: '600',
              }}>
                {statusCounts[filter.key]}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div>
        {filteredExperiments.length === 0 ? (
          <div style={{ 
            padding: '80px 40px', 
            textAlign: 'center', 
            color: theme.textTertiary 
          }}>
            <div style={{ fontSize: '40px', marginBottom: '16px', opacity: 0.5 }}>📋</div>
            <p style={{ margin: 0, fontSize: '15px' }}>
              {experiments.length === 0 
                ? 'No experiments in queue' 
                : `No experiments with status "${statusFilter}"`}
            </p>
          </div>
        ) : (
          filteredExperiments.map((exp) => (
            <div
              key={exp.id}
              style={{
                padding: '28px',
                borderBottom: `1px solid ${theme.borderSubtle}`,
              }}
            >
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
                gap: '28px'
              }}>
                <div style={{ flex: 1 }}>
                  <div style={{
                    fontSize: '15px',
                    fontWeight: '600',
                    marginBottom: '4px',
                    color: theme.textPrimary,
                    fontFamily: "'JetBrains Mono', monospace",
                  }}>
                    {exp.idea_name}
                  </div>

                  {/* Show idea_uid and run_id to distinguish parallel runs */}
                  {(exp.idea_uid || exp.run_id) && (
                    <div style={{
                      fontSize: '12px',
                      color: theme.textTertiary,
                      fontFamily: "'JetBrains Mono', monospace",
                      marginBottom: '8px',
                      display: 'flex',
                      gap: '12px',
                      flexWrap: 'wrap',
                    }}>
                      {exp.idea_uid && (
                        <span>UID: {exp.idea_uid}</span>
                      )}
                      {exp.run_id && (
                        <span>Run: {exp.run_id}</span>
                      )}
                    </div>
                  )}

                  {exp.idea_title && (
                    <div style={{
                      fontSize: '15px',
                      color: theme.textSecondary,
                      marginBottom: '18px',
                      lineHeight: '1.6',
                    }}>
                      {exp.idea_title}
                    </div>
                  )}

                  <div style={{
                    display: 'flex',
                    gap: '18px',
                    alignItems: 'center',
                    flexWrap: 'wrap'
                  }}>
                    <StatusBadge status={exp.status === 'completed' && exp.pgr === null ? 'failed' : exp.status} theme={theme} />

                    {exp.duration_seconds && (
                      <span style={{ 
                        fontSize: '14px', 
                        color: theme.textTertiary,
                        fontFamily: "'JetBrains Mono', monospace",
                      }}>
                        {formatDuration(exp.duration_seconds)}
                      </span>
                    )}

                    {exp.execution_mode && (
                      <span style={{
                        fontSize: '11px',
                        padding: '3px 8px',
                        borderRadius: '4px',
                        background: exp.execution_mode === 'runpod' ? '#EDE9FE' :
                                    exp.execution_mode === 'docker' ? '#DBEAFE' : '#ECFDF5',
                        color: exp.execution_mode === 'runpod' ? '#7C3AED' :
                               exp.execution_mode === 'docker' ? '#2563EB' : '#059669',
                        fontWeight: '500',
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em',
                      }}>
                        {exp.execution_mode}
                      </span>
                    )}

                    {exp.pod_id && exp.status === 'running' && !exp.pod_id.startsWith('local-') && (
                      <a
                        href={`https://console.runpod.io/pods?id=${exp.pod_id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        title="View pod in RunPod console"
                        style={{
                          fontSize: '14px',
                          color: theme.accentCoral,
                          textDecoration: 'none',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '6px',
                        }}
                      >
                        🔗 View Pod
                      </a>
                    )}
                  </div>

                  {/* Timing Info */}
                  <div style={{
                    display: 'flex',
                    gap: '16px',
                    marginTop: '12px',
                    fontSize: '13px',
                    color: theme.textTertiary,
                    flexWrap: 'wrap',
                  }}>
                    {exp.queue_time && (
                      <span title="When added to queue">
                        📥 Queued: {formatDateTime(exp.queue_time)}
                      </span>
                    )}
                    {exp.start_time && (
                      <span title="When experiment started">
                        ▶️ Started: {formatDateTime(exp.start_time)}
                      </span>
                    )}
                    {exp.end_time && (
                      <span title="When experiment finished">
                        ⏹️ Ended: {formatDateTime(exp.end_time)}
                      </span>
                    )}
                  </div>

                  {exp.status === 'completed' && exp.pgr !== null && (
                    <div style={{
                      marginTop: '24px',
                      padding: '20px 24px',
                      background: '#F0FDF4',
                      borderRadius: '10px',
                      border: '1px solid #BBF7D0',
                    }}>
                      <div style={{ 
                        display: 'grid', 
                        gridTemplateColumns: 'repeat(2, 1fr)', 
                        gap: '16px',
                        fontSize: '14px',
                        marginBottom: exp.s3_prefix ? '16px' : '0',
                      }}>
                        <MetricItem 
                          label="PGR" 
                          value={exp.pgr != null ? `${exp.pgr.toFixed(4)}${exp.pgr_se ? ` (±${exp.pgr_se.toFixed(4)})` : ''}` : 'N/A'} 
                          highlight 
                          theme={theme} 
                        />
                        <MetricItem 
                          label="Transfer Acc" 
                          value={exp.transfer_acc != null ? `${exp.transfer_acc.toFixed(4)}${exp.transfer_acc_std ? ` (±${exp.transfer_acc_std.toFixed(4)})` : ''}` : 'N/A'} 
                          highlight 
                          theme={theme} 
                        />
                      </div>
                      {exp.s3_prefix && (
                        <div style={{
                          marginTop: '16px',
                          paddingTop: '16px',
                          borderTop: '1px solid #BBF7D0',
                        }}>
                          <a
                            href={exp.s3_prefix}
                            target="_blank"
                            rel="noopener noreferrer"
                            title="View results in S3"
                            style={{
                              fontSize: '13px',
                              color: theme.accentGreen,
                              textDecoration: 'none',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '6px',
                              fontWeight: '500',
                            }}
                            onMouseEnter={(e) => e.target.style.textDecoration = 'underline'}
                            onMouseLeave={(e) => e.target.style.textDecoration = 'none'}
                          >
                            📦 {exp.s3_prefix}
                          </a>
                        </div>
                      )}
                    </div>
                  )}

                  {exp.status === 'completed' && exp.pgr === null && (
                    <div style={{
                      marginTop: '24px',
                      padding: '20px 24px',
                      background: '#FEF2F2',
                      borderRadius: '10px',
                      border: '1px solid #FECACA',
                    }}>
                      <div style={{
                        fontSize: '14px',
                        color: theme.accentRed,
                        marginBottom: exp.s3_prefix ? '16px' : '0',
                      }}>
                        <strong>Error:</strong> Experiment completed but failed to produce valid results (PGR is N/A)
                      </div>
                      {exp.s3_prefix && (
                        <div style={{
                          marginTop: '16px',
                          paddingTop: '16px',
                          borderTop: '1px solid #FECACA',
                        }}>
                          <a
                            href={exp.s3_prefix}
                            target="_blank"
                            rel="noopener noreferrer"
                            title="View logs in S3"
                            style={{
                              fontSize: '13px',
                              color: theme.accentRed,
                              textDecoration: 'none',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '6px',
                              fontWeight: '500',
                            }}
                            onMouseEnter={(e) => e.target.style.textDecoration = 'underline'}
                            onMouseLeave={(e) => e.target.style.textDecoration = 'none'}
                          >
                            📦 {exp.s3_prefix}
                          </a>
                        </div>
                      )}
                    </div>
                  )}

                  {exp.status === 'failed' && exp.error_msg && (
                    <div style={{
                      marginTop: '24px',
                      padding: '20px 24px',
                      background: '#FEF2F2',
                      borderRadius: '10px',
                      border: '1px solid #FECACA',
                      fontSize: '14px',
                      color: theme.accentRed,
                    }}>
                      <strong>Error:</strong> {exp.error_msg}
                    </div>
                  )}

                  {/* Log buttons */}
                  <div style={{ display: 'flex', gap: '10px', marginTop: '18px', flexWrap: 'wrap' }}>
                    {exp.logs && (
                      <button
                        onClick={() => toggleExpand(exp.id, 'console')}
                        style={{
                          padding: '10px 16px',
                          background: expandedSection[exp.id] === 'console' ? theme.bgElevated : theme.bgTertiary,
                          color: theme.textSecondary,
                          border: 'none',
                          borderRadius: '6px',
                          fontSize: '14px',
                          fontWeight: '500',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease',
                        }}
                      >
                        {expandedSection[exp.id] === 'console' ? '▼' : '▶'} Console Log
                      </button>
                    )}
                    
                    {exp.s3_prefix && (
                      <>
                        <button
                          onClick={() => toggleExpand(exp.id, 's3log')}
                          style={{
                            padding: '10px 16px',
                            background: expandedSection[exp.id] === 's3log' ? '#DBEAFE' : theme.bgTertiary,
                            color: expandedSection[exp.id] === 's3log' ? theme.accentBlue : theme.textSecondary,
                            border: 'none',
                            borderRadius: '6px',
                            fontSize: '14px',
                            fontWeight: '500',
                            cursor: 'pointer',
                            transition: 'all 0.2s ease',
                          }}
                        >
                          {expandedSection[exp.id] === 's3log' ? '▼' : '▶'} Worker Log
                        </button>
                        
                        <button
                          onClick={() => toggleExpand(exp.id, 's3results')}
                          style={{
                            padding: '10px 16px',
                            background: expandedSection[exp.id] === 's3results' ? '#D1FAE5' : theme.bgTertiary,
                            color: expandedSection[exp.id] === 's3results' ? theme.accentGreen : theme.textSecondary,
                            border: 'none',
                            borderRadius: '6px',
                            fontSize: '14px',
                            fontWeight: '500',
                            cursor: 'pointer',
                            transition: 'all 0.2s ease',
                          }}
                        >
                          {expandedSection[exp.id] === 's3results' ? '▼' : '▶'} Results
                        </button>

                        <button
                          onClick={() => toggleExpand(exp.id, 'usage')}
                          style={{
                            padding: '10px 16px',
                            background: expandedSection[exp.id] === 'usage' ? '#FEF3C7' : theme.bgTertiary,
                            color: expandedSection[exp.id] === 'usage' ? theme.accentAmber : theme.textSecondary,
                            border: 'none',
                            borderRadius: '6px',
                            fontSize: '14px',
                            fontWeight: '500',
                            cursor: 'pointer',
                            transition: 'all 0.2s ease',
                          }}
                        >
                          {expandedSection[exp.id] === 'usage' ? '▼' : '▶'} Usage Stats
                        </button>
                      </>
                    )}
                  </div>

                  {/* Console Log */}
                  {expandedSection[exp.id] === 'console' && exp.logs && (
                    <div style={{
                      marginTop: '18px',
                      padding: '20px',
                      background: '#1F2937',
                      borderRadius: '10px',
                      fontSize: '13px',
                      fontFamily: "'JetBrains Mono', monospace",
                      maxHeight: '400px',
                      overflowY: 'auto',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      lineHeight: '1.6',
                      color: '#E5E7EB',
                    }}>
                      {exp.logs}
                    </div>
                  )}
                  
                  {/* S3 Worker Log */}
                  {expandedSection[exp.id] === 's3log' && (
                    <div style={{
                      marginTop: '18px',
                      padding: '20px',
                      background: '#1E3A5F',
                      borderRadius: '10px',
                      fontSize: '13px',
                      fontFamily: "'JetBrains Mono', monospace",
                      maxHeight: '400px',
                      overflowY: 'auto',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      lineHeight: '1.6',
                      color: '#E5E7EB',
                    }}>
                      {s3Data[exp.id]?.loading && (
                        <div style={{ color: '#93C5FD' }}>Loading worker log from S3...</div>
                      )}
                      {s3Data[exp.id]?.error && (
                        <div style={{ color: '#FCA5A5' }}>Error: {s3Data[exp.id].error}</div>
                      )}
                      {s3Data[exp.id]?.log && s3Data[exp.id].log}
                      {s3Data[exp.id] && !s3Data[exp.id].loading && !s3Data[exp.id].error && !s3Data[exp.id].log && (
                        <div style={{ color: '#FCD34D' }}>No worker log found in S3</div>
                      )}
                    </div>
                  )}
                  
                  {/* S3 Results */}
                  {expandedSection[exp.id] === 's3results' && (
                    <div style={{
                      marginTop: '18px',
                      padding: '20px',
                      background: '#14532D',
                      borderRadius: '10px',
                      fontSize: '13px',
                      fontFamily: "'JetBrains Mono', monospace",
                      maxHeight: '400px',
                      overflowY: 'auto',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      lineHeight: '1.6',
                      color: '#E5E7EB',
                    }}>
                      {s3Data[exp.id]?.loading && (
                        <div style={{ color: '#86EFAC' }}>Loading results from S3...</div>
                      )}
                      {s3Data[exp.id]?.error && (
                        <div style={{ color: '#FCA5A5' }}>Error: {s3Data[exp.id].error}</div>
                      )}
                      {s3Data[exp.id]?.results && (
                        <pre style={{ margin: 0 }}>{JSON.stringify(s3Data[exp.id].results, null, 2)}</pre>
                      )}
                      {s3Data[exp.id] && !s3Data[exp.id].loading && !s3Data[exp.id].error && !s3Data[exp.id].results && (
                        <div style={{ color: '#FCD34D' }}>No results found in S3</div>
                      )}
                    </div>
                  )}

                  {/* Usage Stats */}
                  {expandedSection[exp.id] === 'usage' && (
                    <UsageStatsDisplay
                      loading={usageStats[exp.id]?.loading}
                      error={usageStats[exp.id]?.error}
                      stats={usageStats[exp.id]?.stats}
                      theme={theme}
                    />
                  )}
                </div>

                <div style={{ display: 'flex', gap: '12px', flexDirection: 'column' }}>
                  {exp.status === 'queued' && (
                    <ActionButton 
                      onClick={() => onRemove(exp.id)}
                      variant="danger"
                      theme={theme}
                    >
                      Remove
                    </ActionButton>
                  )}

                  {exp.status === 'running' && onKill && (
                    <ActionButton 
                      onClick={() => onKill(exp.id)}
                      variant="danger"
                      theme={theme}
                    >
                      🛑 Kill
                    </ActionButton>
                  )}

                  {(exp.status === 'failed' || exp.status === 'completed') && onRerun && !exp.is_baseline && (
                    <ActionButton 
                      onClick={() => onRerun(exp.id)}
                      variant={(exp.status === 'failed' || (exp.status === 'completed' && exp.pgr === null)) ? 'warning' : 'primary'}
                      theme={theme}
                    >
                      ↻ Rerun
                    </ActionButton>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

const MetricItem = ({ label, value, highlight, theme }) => (
  <div>
    <span style={{ color: theme.textTertiary }}>{label}:</span>
    <span style={{ 
      marginLeft: '8px', 
      fontWeight: '600', 
      color: highlight ? theme.accentGreen : theme.textSecondary,
      fontFamily: "'JetBrains Mono', monospace",
    }}>
      {value || 'N/A'}
    </span>
  </div>
);

// Foldable finding result row for query_findings response display
const FindingResultRow = ({ result, theme }) => {
  const [expanded, setExpanded] = React.useState(false);

  const findingTypeColor = {
    result: '#2E7D5A',
    hypothesis: '#7B5CB5',
    insight: '#B58B3D',
    error: '#DC2626',
    observation: '#4F46E5',
  }[result.finding_type] || '#5C5C5A';

  return (
    <div
      style={{
        marginTop: '3px',
        padding: '4px 8px',
        background: 'rgba(59, 130, 246, 0.05)',
        borderRadius: '4px',
        fontSize: '11px',
        cursor: 'pointer',
        border: '1px solid rgba(59, 130, 246, 0.1)',
      }}
      onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }}
    >
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: '8px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', minWidth: 0 }}>
          <span style={{ color: '#999', fontSize: '10px', flexShrink: 0 }}>
            {expanded ? '▼' : '▶'}
          </span>
          {result.finding_type && (
            <span style={{
              color: findingTypeColor,
              fontWeight: '600',
              fontSize: '9px',
              textTransform: 'uppercase',
              flexShrink: 0,
            }}>
              [{result.finding_type}]
            </span>
          )}
          <span style={{
            fontFamily: "'JetBrains Mono', monospace",
            color: theme.textPrimary,
            fontWeight: '500',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}>
            {result.idea_name || result.title || 'Unknown'}
          </span>
        </div>
        <div style={{ display: 'flex', gap: '8px', flexShrink: 0, alignItems: 'center' }}>
          {result.pgr !== undefined && result.pgr !== null && (
            <span style={{
              color: result.pgr > 0 ? '#2E7D5A' : '#999',
              fontWeight: '500',
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              PGR:{(result.pgr * 100).toFixed(1)}%
            </span>
          )}
          {result.worked !== undefined && result.worked !== null && (
            <span style={{ color: result.worked ? '#2E7D5A' : '#DC2626' }}>
              {result.worked ? '✓' : '✗'}
            </span>
          )}
        </div>
      </div>
      {expanded && (
        <div style={{
          marginTop: '6px',
          paddingTop: '6px',
          borderTop: '1px solid rgba(59, 130, 246, 0.1)',
          lineHeight: '1.5',
        }}>
          {result.title && result.title !== result.idea_name && (
            <div style={{ fontWeight: '500', color: theme.textPrimary, marginBottom: '4px' }}>
              {result.title}
            </div>
          )}
          {result.relevance && (
            <div style={{ color: '#7B5CB5', fontStyle: 'italic', marginBottom: '4px' }}>
              Relevance: {result.relevance}
            </div>
          )}
          {result.content && (
            <div style={{
              color: theme.textSecondary,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              maxHeight: '200px',
              overflowY: 'auto',
              fontSize: '11px',
            }}>
              {result.content}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const UsageStatsDisplay = ({ loading, error, stats, theme }) => {
  if (loading) {
    return (
      <div style={{
        marginTop: '18px',
        padding: '20px',
        background: '#FEF3C7',
        borderRadius: '10px',
        color: theme.accentAmber,
      }}>
        Loading usage statistics...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        marginTop: '18px',
        padding: '20px',
        background: '#FEF3C7',
        borderRadius: '10px',
        color: theme.textTertiary,
      }}>
        {error === 'No usage stats found (experiment may not have used tracked APIs)'
          ? 'No usage stats available for this experiment'
          : `Error: ${error}`}
      </div>
    );
  }

  if (!stats) {
    return (
      <div style={{
        marginTop: '18px',
        padding: '20px',
        background: '#FEF3C7',
        borderRadius: '10px',
        color: theme.textTertiary,
      }}>
        No usage statistics available
      </div>
    );
  }

  const formatDurationMs = (ms) => {
    if (!ms) return 'N/A';
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  const skills = Object.entries(stats.skills || {});
  const mcpTools = Object.entries(stats.mcp_tools || {});
  const summary = stats.summary || {};

  // Aggregate all calls into a single timeline
  const timelineEvents = [];

  // Add skill calls
  skills.forEach(([name, data]) => {
    (data.calls || []).forEach(call => {
      timelineEvents.push({
        type: 'skill',
        name,
        timestamp: call.timestamp,
        duration_ms: call.duration_ms,
        success: call.success,
        session_id: call.metadata?.session_id,
      });
    });
  });

  // Add MCP tool calls
  mcpTools.forEach(([name, data]) => {
    (data.calls || []).forEach(call => {
      timelineEvents.push({
        type: 'mcp',
        name,
        timestamp: call.timestamp,
        duration_ms: call.duration_ms,
        success: call.success,
        session_id: call.metadata?.session_id,
        query_params: call.metadata?.query_params,
        response_results: call.metadata?.response_results,
      });
    });
  });

  // Sort by timestamp (oldest first)
  timelineEvents.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

  // Map session IDs to simple indices (Session 0, Session 1, etc.)
  const uniqueSessionIds = [...new Set(timelineEvents.map(e => e.session_id).filter(Boolean))];
  const sessionIdToIndex = Object.fromEntries(uniqueSessionIds.map((id, idx) => [id, idx]));

  return (
    <div style={{
      marginTop: '18px',
      padding: '20px',
      background: '#FFFBEB',
      borderRadius: '10px',
      border: '1px solid #FCD34D',
    }}>
      {/* Summary */}
      <div style={{
        display: 'flex',
        gap: '24px',
        marginBottom: '16px',
        paddingBottom: '16px',
        borderBottom: '1px solid #FCD34D',
        flexWrap: 'wrap',
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '24px', fontWeight: '700', color: theme.accentAmber }}>
            {summary.total_api_calls || 0}
          </div>
          <div style={{ fontSize: '12px', color: theme.textTertiary }}>Total API Calls</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '24px', fontWeight: '700', color: '#8B5CF6' }}>
            {summary.total_skill_calls || 0}
          </div>
          <div style={{ fontSize: '12px', color: theme.textTertiary }}>Skill Calls</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '24px', fontWeight: '700', color: theme.accentBlue }}>
            {summary.total_mcp_calls || 0}
          </div>
          <div style={{ fontSize: '12px', color: theme.textTertiary }}>MCP Tool Calls</div>
        </div>
      </div>

      {/* Skills Section */}
      {skills.length > 0 && (
        <div style={{ marginBottom: '16px' }}>
          <div style={{
            fontSize: '13px',
            fontWeight: '600',
            color: '#8B5CF6',
            marginBottom: '8px',
          }}>
            Skills
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {skills.map(([name, data]) => (
              <div key={name} style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '8px 12px',
                background: '#F5F3FF',
                borderRadius: '6px',
                fontSize: '13px',
              }}>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", color: theme.textPrimary }}>
                  {name}
                </span>
                <div style={{ display: 'flex', gap: '16px', color: theme.textTertiary }}>
                  <span title="Call count">{data.count}x</span>
                  <span title="Total duration">{formatDurationMs(data.total_duration_ms)}</span>
                  <span title="Success rate" style={{
                    color: data.failures > 0 ? theme.accentRed : theme.accentGreen,
                  }}>
                    {data.successes}/{data.count}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* MCP Tools Section */}
      {mcpTools.length > 0 && (
        <div>
          <div style={{
            fontSize: '13px',
            fontWeight: '600',
            color: theme.accentBlue,
            marginBottom: '8px',
          }}>
            MCP Tools
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {mcpTools.map(([name, data]) => (
              <div key={name} style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '8px 12px',
                background: '#EFF6FF',
                borderRadius: '6px',
                fontSize: '13px',
              }}>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", color: theme.textPrimary }}>
                  {name}
                </span>
                <div style={{ display: 'flex', gap: '16px', color: theme.textTertiary }}>
                  <span title="Call count">{data.count}x</span>
                  <span title="Total duration">{formatDurationMs(data.total_duration_ms)}</span>
                  <span title="Success rate" style={{
                    color: data.failures > 0 ? theme.accentRed : theme.accentGreen,
                  }}>
                    {data.successes}/{data.count}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Timeline Section */}
      {timelineEvents.length > 0 && (
        <div style={{ marginTop: '16px' }}>
          <div style={{
            fontSize: '13px',
            fontWeight: '600',
            color: theme.textSecondary,
            marginBottom: '8px',
          }}>
            Timeline
          </div>
          <div style={{
            maxHeight: '300px',
            overflowY: 'auto',
            paddingLeft: '12px',
            borderLeft: `2px solid ${theme.borderColor || '#E5E7EB'}`,
          }}>
            {timelineEvents.map((event, idx) => (
              <div key={idx} style={{
                position: 'relative',
                paddingLeft: '20px',
                paddingBottom: '16px',
                marginLeft: '-13px',
              }}>
                {/* Timeline dot */}
                <div style={{
                  position: 'absolute',
                  left: 0,
                  top: '4px',
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  background: event.type === 'skill' ? '#8B5CF6' : theme.accentBlue,
                  border: '2px solid white',
                }} />

                {/* Timestamp */}
                <div style={{
                  fontSize: '12px',
                  fontWeight: '600',
                  color: theme.textPrimary,
                  marginBottom: '2px',
                }}>
                  {new Date(event.timestamp).toLocaleTimeString()}
                </div>

                {/* Session ID */}
                {event.session_id && (
                  <div style={{
                    fontSize: '11px',
                    color: theme.textTertiary,
                    marginBottom: '4px',
                  }}>
                    Session {sessionIdToIndex[event.session_id]}
                  </div>
                )}

                {/* Event card */}
                <div style={{
                  padding: '8px 12px',
                  background: event.type === 'skill' ? '#F5F3FF' : '#EFF6FF',
                  borderRadius: '6px',
                  fontSize: '13px',
                }}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}>
                    <span style={{
                      fontFamily: "'JetBrains Mono', monospace",
                      color: theme.textPrimary,
                    }}>
                      {event.type === 'skill' ? '●' : '■'} {event.name}
                    </span>
                    <div style={{ display: 'flex', gap: '12px', color: theme.textTertiary }}>
                      <span>{formatDurationMs(event.duration_ms)}</span>
                      <span style={{ color: event.success ? theme.accentGreen : theme.accentRed }}>
                        {event.success ? '✓' : '✗'}
                      </span>
                    </div>
                  </div>
                  {event.query_params && Object.keys(event.query_params).length > 0 && (
                    <div style={{
                      marginTop: '4px',
                      fontSize: '11px',
                      color: theme.textTertiary,
                      fontFamily: "'JetBrains Mono', monospace",
                      lineHeight: '1.4',
                    }}>
                      {Object.entries(event.query_params).map(([key, value]) => (
                        <span key={key} style={{
                          display: 'inline-block',
                          marginRight: '8px',
                          padding: '1px 5px',
                          background: 'rgba(59, 130, 246, 0.1)',
                          borderRadius: '3px',
                        }}>
                          {key}={typeof value === 'string' ? `"${value}"` : String(value)}
                        </span>
                      ))}
                    </div>
                  )}
                  {event.response_results && event.response_results.length > 0 && (
                    <div style={{
                      marginTop: '6px',
                      fontSize: '11px',
                      color: theme.textTertiary,
                    }}>
                      <div style={{
                        fontWeight: '600',
                        marginBottom: '4px',
                        color: theme.textSecondary,
                      }}>
                        Results ({event.response_results.length}):
                      </div>
                      {event.response_results.map((result, rIdx) => (
                        <FindingResultRow key={rIdx} result={result} theme={theme} />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {skills.length === 0 && mcpTools.length === 0 && (
        <div style={{ color: theme.textTertiary, fontSize: '13px' }}>
          No skill or MCP tool calls recorded yet
        </div>
      )}

      {/* Session info */}
      {stats.session_start && (
        <div style={{
          marginTop: '16px',
          paddingTop: '12px',
          borderTop: '1px solid #FCD34D',
          fontSize: '12px',
          color: theme.textTertiary,
        }}>
          Session started: {new Date(stats.session_start).toLocaleString()}
          {stats.last_updated && (
            <span> | Last updated: {new Date(stats.last_updated).toLocaleString()}</span>
          )}
        </div>
      )}
    </div>
  );
};

const ActionButton = ({ onClick, variant, theme, children }) => {
  const colors = {
    danger: { bg: '#FEE2E2', color: theme.accentRed, border: '#FECACA' },
    warning: { bg: '#FEF3C7', color: '#92400E', border: '#FCD34D' },
    primary: { bg: '#DBEAFE', color: theme.accentBlue, border: '#93C5FD' },
  };
  const c = colors[variant] || colors.primary;

  return (
    <button
      onClick={onClick}
      style={{
        padding: '10px 18px',
        background: c.bg,
        color: c.color,
        border: `1px solid ${c.border}`,
        borderRadius: '8px',
        fontSize: '14px',
        fontWeight: '500',
        cursor: 'pointer',
        whiteSpace: 'nowrap',
        transition: 'all 0.2s ease',
      }}
    >
      {children}
    </button>
  );
};

export default QueuePanel;
