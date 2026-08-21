import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const API_BASE = process.env.REACT_APP_API_BASE || '';

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

const findingTypeColors = {
  hypothesis: { bg: '#EDE9FE', text: '#7B5CB5' },
  result: { bg: '#D1FAE5', text: '#2E7D5A' },
  insight: { bg: '#FEF3C7', text: '#B58B3D' },
  error: { bg: '#FEE2E2', text: '#DC2626' },
  observation: { bg: '#E0E7FF', text: '#4F46E5' },
};

const Forum = ({ theme = defaultTheme }) => {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sort, setSort] = useState('new');
  const [stats, setStats] = useState(null);
  const [selectedPost, setSelectedPost] = useState(null);
  const [newComment, setNewComment] = useState('');
  const [submittingComment, setSubmittingComment] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [searching, setSearching] = useState(false);

  const fetchPosts = useCallback(async (showLoading = true) => {
    try {
      if (showLoading) setLoading(true);
      const response = await axios.get(`${API_BASE}/api/findings`, {
        params: { sort, limit: 100 }
      });
      setPosts(response.data.findings || []);
    } catch (err) {
      console.error('Error fetching forum posts:', err);
    } finally {
      if (showLoading) setLoading(false);
    }
  }, [sort]);

  const fetchStats = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/findings/stats`);
      setStats(response.data);
    } catch (err) {
      console.error('Error fetching forum stats:', err);
    }
  }, []);

  useEffect(() => {
    fetchPosts();
    fetchStats();
  }, [fetchPosts, fetchStats]);

  // Auto-poll every 30 seconds (silent refresh, no loading spinner)
  useEffect(() => {
    const interval = setInterval(() => {
      fetchPosts(false);
      fetchStats();
    }, 30000);
    return () => clearInterval(interval);
  }, [fetchPosts, fetchStats]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchResults(null);
      return;
    }

    setSearching(true);
    try {
      const response = await axios.post(`${API_BASE}/api/findings/search`, {
        query: searchQuery.trim(),
        limit: 20,
      });
      setSearchResults(response.data);
    } catch (err) {
      console.error('Error searching findings:', err);
      setSearchResults({ results: [], summary: 'Search failed.' });
    } finally {
      setSearching(false);
    }
  };

  const handleSearchKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const clearSearch = () => {
    setSearchQuery('');
    setSearchResults(null);
  };

  const handleVote = async (postId, voteType) => {
    try {
      const response = await axios.post(`${API_BASE}/api/findings/${postId}/vote`, {
        vote: voteType
      });
      // Update the post in the list
      setPosts(posts.map(p =>
        p.post_id === postId ? response.data.post : p
      ));
      if (selectedPost && selectedPost.post_id === postId) {
        setSelectedPost(response.data.post);
      }
    } catch (err) {
      console.error('Error voting:', err);
    }
  };

  const handleOpenPost = async (postId) => {
    try {
      const response = await axios.get(`${API_BASE}/api/findings/${postId}`);
      setSelectedPost(response.data.post);
    } catch (err) {
      console.error('Error fetching post details:', err);
    }
  };

  const handleAddComment = async () => {
    if (!newComment.trim() || !selectedPost) return;

    setSubmittingComment(true);
    try {
      await axios.post(`${API_BASE}/api/findings/${selectedPost.post_id}/comments`, {
        content: newComment.trim(),
        author: 'human'
      });
      // Refresh the post to get updated comments
      const response = await axios.get(`${API_BASE}/api/findings/${selectedPost.post_id}`);
      setSelectedPost(response.data.post);
      setNewComment('');
      // Update comment count in list
      setPosts(posts.map(p =>
        p.post_id === selectedPost.post_id
          ? { ...p, comment_count: response.data.post.comment_count }
          : p
      ));
    } catch (err) {
      console.error('Error adding comment:', err);
    } finally {
      setSubmittingComment(false);
    }
  };

  const formatTimeAgo = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const seconds = Math.floor((now - date) / 1000);

    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
    return date.toLocaleDateString();
  };

  if (loading && posts.length === 0) {
    return (
      <div style={{
        padding: '80px 20px',
        textAlign: 'center',
        color: theme.textSecondary,
      }}>
        <div style={{ fontSize: '15px' }}>Loading forum...</div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', gap: '24px', height: '100%' }}>
      {/* Main Feed */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          background: theme.bgSecondary,
          borderRadius: '12px',
          border: `1px solid ${theme.borderSubtle}`,
          overflow: 'hidden'
        }}>
          {/* Header */}
          <div style={{
            padding: '20px 24px',
            borderBottom: `1px solid ${theme.borderSubtle}`,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}>
            <div>
              <h2 style={{
                margin: 0,
                fontSize: '20px',
                fontWeight: '600',
                color: theme.textPrimary,
              }}>
                Agent Forum
              </h2>
              <p style={{
                margin: '4px 0 0 0',
                fontSize: '13px',
                color: theme.textTertiary,
              }}>
                Findings shared by research agents
              </p>
            </div>

            {/* Sort Tabs */}
            <div style={{ display: 'flex', gap: '4px', background: theme.bgTertiary, padding: '4px', borderRadius: '8px' }}>
              {['new', 'top', 'discussed'].map(s => (
                <button
                  key={s}
                  onClick={() => setSort(s)}
                  style={{
                    padding: '6px 14px',
                    fontSize: '13px',
                    fontWeight: sort === s ? '600' : '500',
                    color: sort === s ? theme.textPrimary : theme.textSecondary,
                    background: sort === s ? theme.bgSecondary : 'transparent',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    textTransform: 'capitalize',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          {/* Search Bar */}
          <div style={{
            padding: '12px 24px',
            borderBottom: `1px solid ${theme.borderSubtle}`,
            display: 'flex',
            gap: '8px',
            alignItems: 'center',
          }}>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={handleSearchKeyDown}
              placeholder="Semantic search findings... (e.g., 'methods that reduce label noise')"
              style={{
                flex: 1,
                padding: '8px 12px',
                fontSize: '13px',
                border: `1px solid ${theme.borderDefault}`,
                borderRadius: '6px',
                background: theme.bgPrimary,
                color: theme.textPrimary,
                outline: 'none',
              }}
            />
            <button
              onClick={handleSearch}
              disabled={searching || !searchQuery.trim()}
              style={{
                padding: '8px 16px',
                fontSize: '13px',
                fontWeight: '500',
                color: '#fff',
                background: searchQuery.trim() ? theme.accentPurple : theme.textTertiary,
                border: 'none',
                borderRadius: '6px',
                cursor: searchQuery.trim() ? 'pointer' : 'not-allowed',
                transition: 'background 0.15s ease',
                whiteSpace: 'nowrap',
              }}
            >
              {searching ? 'Searching...' : 'Search'}
            </button>
            {searchResults && (
              <button
                onClick={clearSearch}
                style={{
                  padding: '8px 12px',
                  fontSize: '13px',
                  color: theme.textSecondary,
                  background: theme.bgTertiary,
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                }}
              >
                Clear
              </button>
            )}
          </div>

          {/* Search Results */}
          {searchResults && (
            <div style={{
              padding: '16px 24px',
              borderBottom: `1px solid ${theme.borderSubtle}`,
              background: '#F8F7FF',
            }}>
              <div style={{
                fontSize: '12px',
                fontWeight: '600',
                color: theme.accentPurple,
                textTransform: 'uppercase',
                marginBottom: '8px',
              }}>
                Semantic Search Results
              </div>
              {searchResults.summary && (
                <p style={{
                  margin: '0 0 12px 0',
                  fontSize: '13px',
                  color: theme.textSecondary,
                  lineHeight: 1.5,
                }}>
                  {searchResults.summary}
                </p>
              )}
              {searchResults.results && searchResults.results.length > 0 ? (
                searchResults.results.map((r, i) => (
                  <div key={i} style={{
                    padding: '10px 12px',
                    marginBottom: '8px',
                    background: theme.bgSecondary,
                    borderRadius: '6px',
                    border: `1px solid ${theme.borderSubtle}`,
                  }}>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                      {r.finding_type && (
                        <span style={{
                          fontSize: '10px',
                          fontWeight: '600',
                          color: (findingTypeColors[r.finding_type] || findingTypeColors.observation).text,
                          background: (findingTypeColors[r.finding_type] || findingTypeColors.observation).bg,
                          padding: '2px 6px',
                          borderRadius: '3px',
                          textTransform: 'uppercase',
                        }}>
                          {r.finding_type}
                        </span>
                      )}
                      <span style={{ fontSize: '14px', fontWeight: '600', color: theme.textPrimary }}>
                        {r.title}
                      </span>
                      {r.pgr !== null && r.pgr !== undefined && (
                        <span style={{ fontSize: '12px', color: theme.accentGreen, fontWeight: '500' }}>
                          PGR: {(r.pgr * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    {r.idea_name && (
                      <div style={{ fontSize: '12px', color: theme.textTertiary, marginTop: '4px', fontFamily: "'JetBrains Mono', monospace" }}>
                        {r.idea_name}
                      </div>
                    )}
                    <div style={{ fontSize: '12px', color: theme.textSecondary, marginTop: '4px', fontStyle: 'italic' }}>
                      {r.relevance}
                    </div>
                  </div>
                ))
              ) : (
                <p style={{ margin: 0, fontSize: '13px', color: theme.textTertiary }}>
                  No semantically relevant findings found.
                </p>
              )}
            </div>
          )}

          {/* Posts List */}
          <div style={{ maxHeight: '600px', overflowY: 'auto' }}>
            {posts.length === 0 ? (
              <div style={{
                padding: '80px 40px',
                textAlign: 'center',
                color: theme.textTertiary
              }}>
                <div style={{ fontSize: '48px', marginBottom: '16px', opacity: 0.5 }}>💬</div>
                <p style={{ margin: 0, fontSize: '15px' }}>No posts yet</p>
                <p style={{ margin: '8px 0 0 0', fontSize: '13px' }}>
                  Agent findings will appear here when they share their research
                </p>
              </div>
            ) : (
              posts.map(post => (
                <div
                  key={post.post_id}
                  onClick={() => handleOpenPost(post.post_id)}
                  style={{
                    padding: '16px 24px',
                    borderBottom: `1px solid ${theme.borderSubtle}`,
                    cursor: 'pointer',
                    transition: 'background 0.15s ease',
                    background: selectedPost?.post_id === post.post_id ? theme.bgTertiary : 'transparent',
                  }}
                  onMouseOver={(e) => {
                    if (selectedPost?.post_id !== post.post_id) {
                      e.currentTarget.style.background = theme.bgTertiary;
                    }
                  }}
                  onMouseOut={(e) => {
                    if (selectedPost?.post_id !== post.post_id) {
                      e.currentTarget.style.background = 'transparent';
                    }
                  }}
                >
                  <div style={{ display: 'flex', gap: '16px' }}>
                    {/* Vote Column */}
                    <div style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: '4px',
                      minWidth: '40px',
                    }}>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleVote(post.post_id, 'up'); }}
                        style={{
                          background: 'none',
                          border: 'none',
                          cursor: 'pointer',
                          fontSize: '16px',
                          padding: '4px',
                          opacity: 0.6,
                          transition: 'opacity 0.15s',
                        }}
                        onMouseOver={(e) => e.target.style.opacity = 1}
                        onMouseOut={(e) => e.target.style.opacity = 0.6}
                      >
                        ▲
                      </button>
                      <span style={{
                        fontSize: '14px',
                        fontWeight: '600',
                        color: post.score > 0 ? theme.accentGreen : post.score < 0 ? theme.accentCoral : theme.textSecondary,
                      }}>
                        {post.score}
                      </span>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleVote(post.post_id, 'down'); }}
                        style={{
                          background: 'none',
                          border: 'none',
                          cursor: 'pointer',
                          fontSize: '16px',
                          padding: '4px',
                          opacity: 0.6,
                          transition: 'opacity 0.15s',
                        }}
                        onMouseOver={(e) => e.target.style.opacity = 1}
                        onMouseOut={(e) => e.target.style.opacity = 0.6}
                      >
                        ▼
                      </button>
                    </div>

                    {/* Content */}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                        {post.finding_type && (
                          <span style={{
                            fontSize: '10px',
                            fontWeight: '600',
                            color: (findingTypeColors[post.finding_type] || findingTypeColors.observation).text,
                            background: (findingTypeColors[post.finding_type] || findingTypeColors.observation).bg,
                            padding: '3px 8px',
                            borderRadius: '4px',
                            textTransform: 'uppercase',
                          }}>
                            {post.finding_type}
                          </span>
                        )}
                        <h3 style={{
                          margin: 0,
                          fontSize: '15px',
                          fontWeight: '600',
                          color: theme.textPrimary,
                          lineHeight: 1.4,
                        }}>
                          {post.title}
                        </h3>
                      </div>

                      <p style={{
                        margin: '8px 0 0 0',
                        fontSize: '13px',
                        color: theme.textSecondary,
                        lineHeight: 1.5,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                      }}>
                        {post.content}
                      </p>

                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        marginTop: '10px',
                        fontSize: '12px',
                        color: theme.textTertiary,
                        flexWrap: 'wrap',
                      }}>
                        {post.idea_name && (
                          <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                            {post.idea_name}
                          </span>
                        )}
                        {post.pgr !== null && (
                          <span style={{
                            color: post.pgr > 0 ? theme.accentGreen : theme.textTertiary,
                            fontWeight: '500',
                          }}>
                            PGR: {(post.pgr * 100).toFixed(1)}%
                          </span>
                        )}
                        {post.dataset && (
                          <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                            {post.dataset}
                          </span>
                        )}
                        {post.weak_model && (
                          <span style={{ fontFamily: "'JetBrains Mono', monospace", opacity: 0.8 }}>
                            weak: {post.weak_model}
                          </span>
                        )}
                        {post.strong_model && (
                          <span style={{ fontFamily: "'JetBrains Mono', monospace", opacity: 0.8 }}>
                            strong: {post.strong_model}
                          </span>
                        )}
                        <span>{post.comment_count} comments</span>
                        <span>{formatTimeAgo(post.created_at)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <div style={{ width: '420px', flexShrink: 0 }}>
        {/* Stats Card */}
        {stats && (
          <div style={{
            background: theme.bgSecondary,
            borderRadius: '12px',
            border: `1px solid ${theme.borderSubtle}`,
            padding: '20px',
            marginBottom: '16px',
          }}>
            <h3 style={{
              margin: '0 0 16px 0',
              fontSize: '14px',
              fontWeight: '600',
              color: theme.textPrimary,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}>
              Forum Stats
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <StatItem label="Total Posts" value={stats.total_posts} theme={theme} />
              <StatItem label="Total Comments" value={stats.total_comments} theme={theme} />
              <StatItem label="Total Upvotes" value={stats.total_upvotes} theme={theme} />
            </div>
          </div>
        )}

        {/* Selected Post Detail */}
        {selectedPost && (
          <div style={{
            background: theme.bgSecondary,
            borderRadius: '12px',
            border: `1px solid ${theme.borderSubtle}`,
            overflow: 'hidden',
          }}>
            <div style={{
              padding: '16px 20px',
              borderBottom: `1px solid ${theme.borderSubtle}`,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}>
              <h3 style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: theme.textPrimary }}>
                Post Details
              </h3>
              <button
                onClick={() => setSelectedPost(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '18px',
                  color: theme.textTertiary,
                  padding: '0',
                }}
              >
                ×
              </button>
            </div>

            <div style={{ padding: '16px 20px', maxHeight: '600px', overflowY: 'auto' }}>
              <h4 style={{ margin: '0 0 8px 0', fontSize: '15px', color: theme.textPrimary }}>
                {selectedPost.title}
              </h4>
              <div style={{
                margin: '0 0 16px 0',
                fontSize: '13px',
                color: theme.textSecondary,
                lineHeight: 1.6,
              }}>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    table: ({node, ...props}) => (
                      <table style={{
                        borderCollapse: 'collapse',
                        width: '100%',
                        margin: '8px 0',
                        fontSize: '12px',
                      }} {...props} />
                    ),
                    th: ({node, ...props}) => (
                      <th style={{
                        border: `1px solid ${theme.bgElevated}`,
                        padding: '6px 10px',
                        textAlign: 'left',
                        background: theme.bgTertiary,
                        fontWeight: '600',
                        color: theme.textPrimary,
                      }} {...props} />
                    ),
                    td: ({node, ...props}) => (
                      <td style={{
                        border: `1px solid ${theme.bgElevated}`,
                        padding: '6px 10px',
                        color: theme.textSecondary,
                      }} {...props} />
                    ),
                    pre: ({node, ...props}) => (
                      <pre style={{
                        background: '#1e1e1e',
                        color: '#d4d4d4',
                        padding: '12px',
                        borderRadius: '6px',
                        fontSize: '11px',
                        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                        lineHeight: 1.5,
                        overflow: 'auto',
                        maxHeight: '300px',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                      }} {...props} />
                    ),
                    code: ({node, className, children, ...props}) => {
                      const isInline = !className;
                      return isInline
                        ? <code style={{
                            background: theme.bgTertiary,
                            padding: '1px 4px',
                            borderRadius: '3px',
                            fontSize: '12px',
                            fontFamily: "'JetBrains Mono', monospace",
                          }} {...props}>{children}</code>
                        : <code className={className} {...props}>{children}</code>;
                    },
                    h3: ({node, ...props}) => (
                      <h3 style={{
                        fontSize: '13px',
                        fontWeight: '600',
                        color: theme.textPrimary,
                        margin: '16px 0 8px 0',
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em',
                      }} {...props} />
                    ),
                  }}
                >
                  {selectedPost.content}
                </ReactMarkdown>
              </div>

              {/* Comments */}
              {selectedPost.comments && selectedPost.comments.length > 0 && (
                <div style={{ borderTop: `1px solid ${theme.borderSubtle}`, paddingTop: '16px', marginTop: '16px' }}>
                  <h5 style={{ margin: '0 0 12px 0', fontSize: '12px', color: theme.textTertiary, textTransform: 'uppercase' }}>
                    Comments ({selectedPost.comments.length})
                  </h5>
                  {selectedPost.comments.map(comment => (
                    <div key={comment.id} style={{
                      marginBottom: '12px',
                      padding: '10px 12px',
                      background: theme.bgTertiary,
                      borderRadius: '6px',
                    }}>
                      <div style={{ fontSize: '12px', color: theme.textTertiary, marginBottom: '4px' }}>
                        {comment.author || 'anonymous'} · {formatTimeAgo(comment.created_at)}
                      </div>
                      <p style={{ margin: 0, fontSize: '13px', color: theme.textPrimary, lineHeight: 1.5 }}>
                        {comment.content}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {/* Add Comment */}
              <div style={{ marginTop: '16px' }}>
                <textarea
                  value={newComment}
                  onChange={(e) => setNewComment(e.target.value)}
                  placeholder="Add a comment..."
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    fontSize: '13px',
                    border: `1px solid ${theme.borderDefault}`,
                    borderRadius: '6px',
                    resize: 'vertical',
                    minHeight: '60px',
                    fontFamily: 'inherit',
                    background: theme.bgPrimary,
                    color: theme.textPrimary,
                    boxSizing: 'border-box',
                  }}
                />
                <button
                  onClick={handleAddComment}
                  disabled={!newComment.trim() || submittingComment}
                  style={{
                    marginTop: '8px',
                    padding: '8px 16px',
                    fontSize: '13px',
                    fontWeight: '500',
                    color: '#fff',
                    background: newComment.trim() ? theme.accentGreen : theme.textTertiary,
                    border: 'none',
                    borderRadius: '6px',
                    cursor: newComment.trim() ? 'pointer' : 'not-allowed',
                    transition: 'background 0.15s ease',
                  }}
                >
                  {submittingComment ? 'Posting...' : 'Post Comment'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const StatItem = ({ label, value, theme }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
    <span style={{ fontSize: '13px', color: theme.textSecondary }}>{label}</span>
    <span style={{ fontSize: '15px', fontWeight: '600', color: theme.textPrimary }}>{value}</span>
  </div>
);

export default Forum;
