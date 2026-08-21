import React from 'react';

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
  accentPurple: '#7B5CB5',
  accentAmber: '#B58B3D',
};

const Leaderboard = ({ experiments, loading, theme = defaultTheme }) => {
  if (loading) {
    return (
      <div style={{ 
        padding: '80px 20px', 
        textAlign: 'center',
        color: theme.textSecondary,
      }}>
        <div style={{ fontSize: '15px' }}>Loading leaderboard...</div>
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
        }}>
          Leaderboard
        </h2>
      </div>

      <div>
        {experiments.length === 0 ? (
          <div style={{ 
            padding: '80px 40px', 
            textAlign: 'center', 
            color: theme.textTertiary 
          }}>
            <div style={{ fontSize: '40px', marginBottom: '16px', opacity: 0.5 }}>🏆</div>
            <p style={{ margin: 0, fontSize: '15px' }}>No completed experiments yet</p>
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
            <thead>
              <tr style={{
                borderBottom: `1px solid ${theme.borderSubtle}`,
                background: theme.bgTertiary,
              }}>
                <TableHeader align="center" width="70px" theme={theme}>Rank</TableHeader>
                <TableHeader align="left" width="auto" theme={theme}>Idea</TableHeader>
                <TableHeader align="right" width="180px" theme={theme}>PGR (Std Error)</TableHeader>
                <TableHeader align="right" width="200px" theme={theme}>Transfer (Std Error)</TableHeader>
                <TableHeader align="right" width="110px" theme={theme}>Weak Acc</TableHeader>
                <TableHeader align="right" width="130px" theme={theme}>Strong Acc</TableHeader>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp, index) => {
                const isTopThree = index < 3;
                const rankBgs = ['#FEF3C7', '#F3F4F6', '#FFEDD5'];
                const rankEmojis = ['🥇', '🥈', '🥉'];

                return (
                  <tr
                    key={exp.id}
                    style={{
                      borderBottom: `1px solid ${theme.borderSubtle}`,
                      transition: 'background 0.2s ease',
                    }}
                    onMouseOver={(e) => {
                      e.currentTarget.style.background = theme.bgTertiary;
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.background = 'transparent';
                    }}
                  >
                    <td style={{
                      padding: '16px 12px',
                      textAlign: 'center',
                      background: isTopThree ? rankBgs[index] : 'transparent',
                      fontWeight: isTopThree ? '700' : '500',
                      fontSize: isTopThree ? '18px' : '14px',
                      color: theme.textPrimary,
                    }}>
                      {isTopThree ? rankEmojis[index] : `#${index + 1}`}
                    </td>
                    <td style={{ padding: '16px 20px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                        <div style={{
                          fontSize: '15px',
                          fontWeight: '600',
                          color: theme.textPrimary,
                          fontFamily: "'JetBrains Mono', monospace",
                        }}>
                          {exp.idea_name}
                        </div>
                        {exp.is_baseline && (
                          <span style={{
                            fontSize: '10px',
                            fontWeight: '600',
                            color: theme.accentPurple,
                            background: '#EDE9FE',
                            padding: '4px 10px',
                            borderRadius: '5px',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em',
                          }}>
                            Baseline
                          </span>
                        )}
                        {exp.num_seeds && (
                          <span style={{
                            fontSize: '10px',
                            fontWeight: '500',
                            color: theme.textTertiary,
                            background: theme.bgTertiary,
                            padding: '4px 8px',
                            borderRadius: '4px',
                          }}>
                            {exp.num_seeds} seeds
                          </span>
                        )}
                      </div>
                      {exp.idea_title && (
                        <div style={{
                          fontSize: '14px',
                          color: theme.textTertiary,
                          marginTop: '8px',
                          lineHeight: '1.5',
                        }}>
                          {exp.idea_title}
                        </div>
                      )}
                    </td>
                    <td style={{
                      padding: '16px 16px',
                      textAlign: 'right',
                      fontSize: '14px',
                      fontWeight: '700',
                      color: theme.accentGreen,
                      fontFamily: "'JetBrains Mono', monospace",
                    }}>
                      {exp.pgr != null ? (
                        <span>
                          {exp.pgr.toFixed(4)}
                          {exp.pgr_se != null && (
                            <span style={{ fontWeight: '400', color: theme.textTertiary, marginLeft: '4px' }}>
                              (±{exp.pgr_se.toFixed(4)})
                            </span>
                          )}
                        </span>
                      ) : 'N/A'}
                    </td>
                    <td style={{
                      padding: '16px 16px',
                      textAlign: 'right',
                      fontSize: '14px',
                      fontWeight: '500',
                      color: theme.textSecondary,
                      fontFamily: "'JetBrains Mono', monospace",
                    }}>
                      {exp.transfer_acc != null ? (
                        <span>
                          {exp.transfer_acc.toFixed(4)}
                          {(exp.transfer_acc_se || exp.transfer_acc_std) && (
                            <span style={{ fontWeight: '400', color: theme.textTertiary, marginLeft: '4px' }}>
                              (±{(exp.transfer_acc_se || exp.transfer_acc_std).toFixed(4)})
                            </span>
                          )}
                        </span>
                      ) : 'N/A'}
                    </td>
                    <td style={{
                      padding: '16px 16px',
                      textAlign: 'right',
                      fontSize: '14px',
                      fontWeight: '500',
                      color: theme.textTertiary,
                      fontFamily: "'JetBrains Mono', monospace",
                    }}>
                      {exp.weak_acc?.toFixed(4) || 'N/A'}
                    </td>
                    <td style={{
                      padding: '16px 16px',
                      textAlign: 'right',
                      fontSize: '14px',
                      fontWeight: '500',
                      color: theme.textTertiary,
                      fontFamily: "'JetBrains Mono', monospace",
                    }}>
                      {exp.strong_acc?.toFixed(4) || 'N/A'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

const TableHeader = ({ children, align = 'left', width, theme }) => (
  <th style={{
    padding: '16px 16px',
    textAlign: align,
    fontSize: '11px',
    fontWeight: '600',
    color: theme.textTertiary,
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    width,
  }}>
    {children}
  </th>
);

export default Leaderboard;
