import React from 'react';

// Anthropic light theme status colors
const statusConfig = {
  queued: {
    bg: '#FEF3C7',
    color: '#92400E',
    label: 'Queued'
  },
  running: {
    bg: '#DBEAFE',
    color: '#1E40AF',
    label: 'Running'
  },
  completed: {
    bg: '#D1FAE5',
    color: '#065F46',
    label: 'Completed'
  },
  failed: {
    bg: '#FEE2E2',
    color: '#991B1B',
    label: 'Failed'
  },
  baseline: {
    bg: '#EDE9FE',
    color: '#5B21B6',
    label: 'Baseline'
  }
};

const StatusBadge = ({ status, theme }) => {
  const config = statusConfig[status] || statusConfig.queued;

  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: '6px',
      padding: '5px 12px',
      borderRadius: '6px',
      fontSize: '12px',
      fontWeight: '600',
      textTransform: 'uppercase',
      letterSpacing: '0.04em',
      background: config.bg,
      color: config.color,
    }}>
      {status === 'running' && (
        <span style={{
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          background: config.color,
          animation: 'pulse 1.5s ease-in-out infinite',
        }} />
      )}
      {config.label}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </span>
  );
};

export default StatusBadge;
