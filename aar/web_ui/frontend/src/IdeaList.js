import React, { useState, useMemo } from 'react';
import StatusBadge from './StatusBadge';
import AddIdeaForm from './AddIdeaForm';

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
  accentAmber: '#B58B3D',
  accentBlue: '#4B7CC5',
  accentRed: '#C54B4B',
};

const IdeaList = ({ ideas, onAddToQueue, loading, onIdeaAdded, theme = defaultTheme, dataset, weakModel, strongModel }) => {
  const [expandedIdeas, setExpandedIdeas] = useState(new Set());
  const [showAddForm, setShowAddForm] = useState(false);
  const [tagFilter, setTagFilter] = useState('all'); // 'all', 'baseline', 'seed', 'generated'

  // Filter and sort ideas by tag
  // Order: Baselines first, then Seeds, then Generated (within each group: reverse order for newest first)
  const filteredIdeas = useMemo(() => {
    let filtered = [...ideas];

    // Apply tag filter
    if (tagFilter === 'baseline') {
      filtered = filtered.filter(idea => idea.is_baseline);
    } else if (tagFilter === 'seed') {
      filtered = filtered.filter(idea => idea.is_seed && !idea.is_baseline);
    } else if (tagFilter === 'generated') {
      filtered = filtered.filter(idea => !idea.is_baseline && !idea.is_seed);
    }

    // Sort: Baselines first, then non-baseline seeds, then generated, then others
    // Within each group, maintain reverse order (newest first)
    filtered.sort((a, b) => {
      // Priority: baseline (0) > seed-only (1) > generated (2) > other (3)
      const getPriority = (idea) => {
        if (idea.is_baseline) return 0;
        if (idea.is_seed) return 1;
        return 2;
      };
      return getPriority(a) - getPriority(b);
    });

    return filtered;
  }, [ideas, tagFilter]);

  // Count ideas by tag
  const tagCounts = useMemo(() => {
    return {
      all: ideas.length,
      baseline: ideas.filter(i => i.is_baseline).length,
      seed: ideas.filter(i => i.is_seed && !i.is_baseline).length,
      generated: ideas.filter(i => !i.is_baseline && !i.is_seed).length,
    };
  }, [ideas]);

  const toggleExpand = (index) => {
    const newExpanded = new Set(expandedIdeas);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedIdeas(newExpanded);
  };

  if (loading) {
    return (
      <div style={{ 
        padding: '80px 20px', 
        textAlign: 'center',
        color: theme.textSecondary,
      }}>
        <div style={{ fontSize: '15px' }}>Loading ideas...</div>
      </div>
    );
  }

  const handleIdeaAdded = (response) => {
    setShowAddForm(false);
    if (onIdeaAdded) {
      onIdeaAdded(response);
    }
  };

  return (
    <div>
      {/* Add Idea Form */}
      {showAddForm && (
        <AddIdeaForm
          onIdeaAdded={handleIdeaAdded}
          onCancel={() => setShowAddForm(false)}
          theme={theme}
          dataset={dataset}
          weakModel={weakModel}
          strongModel={strongModel}
        />
      )}

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
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '20px',
          }}>
            <h2 style={{ 
              margin: 0, 
              fontSize: '20px', 
              fontWeight: '600',
              color: theme.textPrimary,
            }}>
              Available Ideas
            </h2>
            <button
              onClick={() => setShowAddForm(!showAddForm)}
              style={{
                padding: '10px 20px',
                background: showAddForm ? theme.bgTertiary : theme.accentCoral,
                color: showAddForm ? theme.textSecondary : '#FFFFFF',
                border: 'none',
                borderRadius: '8px',
                fontSize: '14px',
                fontWeight: '500',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
            >
              {showAddForm ? 'Cancel' : '+ Add Idea'}
            </button>
          </div>
          
          {/* Tag Filter */}
          <div style={{
            display: 'flex',
            gap: '8px',
            flexWrap: 'wrap',
          }}>
            {[
              { key: 'all', label: 'All', color: theme.textPrimary },
              { key: 'baseline', label: 'Baseline', color: '#8B5CF6' },
              { key: 'seed', label: 'Seed', color: theme.accentBlue },
              { key: 'generated', label: 'Generated', color: theme.accentGreen },
            ].map(filter => (
              <button
                key={filter.key}
                onClick={() => setTagFilter(filter.key)}
                style={{
                  padding: '6px 14px',
                  background: tagFilter === filter.key ? `${filter.color}15` : theme.bgTertiary,
                  color: tagFilter === filter.key ? filter.color : theme.textTertiary,
                  border: tagFilter === filter.key ? `1px solid ${filter.color}40` : `1px solid ${theme.borderSubtle}`,
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
                  background: tagFilter === filter.key ? `${filter.color}25` : theme.bgElevated,
                  padding: '2px 8px',
                  borderRadius: '10px',
                  fontSize: '12px',
                  fontWeight: '600',
                }}>
                  {tagCounts[filter.key]}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div>
          {filteredIdeas.length === 0 ? (
            <div style={{ 
              padding: '80px 40px', 
              textAlign: 'center', 
              color: theme.textTertiary 
            }}>
              <div style={{ fontSize: '40px', marginBottom: '16px', opacity: 0.5 }}>📭</div>
              <p style={{ margin: 0, fontSize: '15px' }}>
                {ideas.length === 0 
                  ? 'No ideas available' 
                  : `No ${tagFilter} ideas found`}
              </p>
            </div>
          ) : (
            filteredIdeas.map((idea, index) => {
              // Use idea Name as unique key for expansion state
              const ideaKey = idea.Name || index;
              const isExpanded = expandedIdeas.has(ideaKey);

              return (
                <div
                  key={ideaKey}
                  style={{
                    borderBottom: `1px solid ${theme.borderSubtle}`,
                  }}
                >
                  <div
                    style={{
                      padding: '24px 28px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'flex-start',
                      gap: '24px',
                      cursor: 'pointer',
                      background: isExpanded ? theme.bgTertiary : 'transparent',
                      transition: 'background 0.2s ease',
                    }}
                    onClick={() => toggleExpand(ideaKey)}
                  >
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <span style={{
                          fontSize: '10px',
                          color: theme.textTertiary,
                          transition: 'transform 0.2s ease',
                          transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                        }}>
                          ▶
                        </span>
                        <div style={{
                          fontSize: '15px',
                          fontWeight: '600',
                          color: theme.textPrimary,
                          fontFamily: "'JetBrains Mono', monospace",
                        }}>
                          {idea.Name}
                        </div>
                        {/* Baseline/Seed tags */}
                        {idea.is_baseline && (
                          <span style={{
                            padding: '3px 8px',
                            background: '#8B5CF620',
                            color: '#8B5CF6',
                            borderRadius: '4px',
                            fontSize: '11px',
                            fontWeight: '600',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em',
                          }}>
                            Baseline
                          </span>
                        )}
                        {idea.is_seed && (
                          <span style={{
                            padding: '3px 8px',
                            background: `${theme.accentBlue}20`,
                            color: theme.accentBlue,
                            borderRadius: '4px',
                            fontSize: '11px',
                            fontWeight: '600',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em',
                          }}>
                            Seed
                          </span>
                        )}
                      </div>
                      {idea.in_queue && (
                        <div style={{ marginTop: '14px', marginLeft: '22px', display: 'flex', alignItems: 'center', gap: '14px' }}>
                          <StatusBadge status={idea.queue_status} theme={theme} />
                          {idea.pgr !== null && idea.pgr !== undefined && (
                            <span style={{
                              fontSize: '14px',
                              color: theme.accentGreen,
                              fontWeight: '600',
                              fontFamily: "'JetBrains Mono', monospace",
                            }}>
                              PGR: {idea.pgr.toFixed(4)}
                            </span>
                          )}
                        </div>
                      )}
                    </div>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onAddToQueue(idea);
                      }}
                      disabled={idea.is_baseline}
                      title={idea.is_baseline ? 'Baseline ideas use pre-computed results' : 'Queue this idea (can be queued multiple times for parallel runs)'}
                      style={{
                        padding: '10px 18px',
                        background: idea.is_baseline ? theme.bgElevated : theme.accentCoral,
                        color: idea.is_baseline ? theme.textTertiary : '#FFFFFF',
                        border: 'none',
                        borderRadius: '8px',
                        fontSize: '14px',
                        fontWeight: '500',
                        cursor: idea.is_baseline ? 'not-allowed' : 'pointer',
                        whiteSpace: 'nowrap',
                        transition: 'all 0.2s ease',
                      }}
                    >
                      {idea.is_baseline ? 'Pre-computed' : 'Add to Queue'}
                    </button>
                  </div>

                  {/* Expanded Details */}
                  {isExpanded && (
                    <div style={{
                      padding: '28px 28px 32px 50px',
                      background: theme.bgTertiary,
                      borderTop: `1px solid ${theme.borderSubtle}`,
                    }}>
                      {idea.Description && (
                        <Section title="Description" theme={theme} preWrap>
                          {idea.Description}
                        </Section>
                      )}

                      {idea.uid && (
                        <Section title="Idea UID" theme={theme}>
                          <code style={{ 
                            fontFamily: "'JetBrains Mono', monospace",
                            background: theme.bgElevated,
                            padding: '4px 8px',
                            borderRadius: '4px',
                            fontSize: '13px',
                          }}>
                            {idea.uid}
                          </code>
                        </Section>
                      )}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};

const Section = ({ title, children, theme, preWrap }) => (
  <div style={{ marginBottom: '28px' }}>
    <div style={{
      fontSize: '11px',
      fontWeight: '600',
      color: theme.textTertiary,
      marginBottom: '10px',
      textTransform: 'uppercase',
      letterSpacing: '0.1em',
    }}>
      {title}
    </div>
    <div style={{
      fontSize: '15px',
      color: theme.textSecondary,
      lineHeight: '1.7',
      whiteSpace: preWrap ? 'pre-wrap' : 'normal',
    }}>
      {children}
    </div>
  </div>
);

export default IdeaList;
