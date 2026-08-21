import React, { useState } from 'react';
import axios from 'axios';

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
  accentRed: '#C54B4B',
};

const AddIdeaForm = ({ onIdeaAdded, onCancel, theme = defaultTheme, dataset, weakModel, strongModel }) => {
  const [formData, setFormData] = useState({
    Name: '',
    Description: '',
  });
  const [addToQueue, setAddToQueue] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);

    try {
      const response = await axios.post(`${API_BASE}/api/ideas`, {
        idea: formData,
        add_to_queue: addToQueue,
        dataset: dataset,
        weak_model: weakModel,
        strong_model: strongModel,
      });

      setFormData({
        Name: '',
        Description: '',
      });
      setAddToQueue(false);

      if (onIdeaAdded) {
        onIdeaAdded(response.data);
      }
    } catch (err) {
      console.error('Error creating idea:', err);
      setError(err.response?.data?.error || err.message || 'Failed to create idea');
    } finally {
      setSubmitting(false);
    }
  };

  const inputStyle = {
    width: '100%',
    padding: '12px 14px',
    background: theme.bgSecondary,
    border: `1px solid ${theme.borderDefault}`,
    borderRadius: '8px',
    fontSize: '15px',
    fontFamily: 'inherit',
    color: theme.textPrimary,
    transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
  };

  const labelStyle = {
    display: 'block',
    fontSize: '14px',
    fontWeight: '500',
    color: theme.textPrimary,
    marginBottom: '8px',
  };

  return (
    <div style={{
      background: theme.bgSecondary,
      borderRadius: '12px',
      border: `1px solid ${theme.borderSubtle}`,
      padding: '32px',
      marginBottom: '28px'
    }}>
      <h2 style={{
        margin: '0 0 28px 0',
        fontSize: '20px',
        fontWeight: '600',
        color: theme.textPrimary,
      }}>
        Add New Idea
      </h2>

      {error && (
        <div style={{
          background: '#FEF2F2',
          color: theme.accentRed,
          padding: '14px 18px',
          borderRadius: '8px',
          marginBottom: '24px',
          fontSize: '14px',
          border: `1px solid ${theme.accentRed}30`,
        }}>
          ⚠ {error}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div style={{ display: 'grid', gap: '24px' }}>
          {/* Name - Required */}
          <div>
            <label style={labelStyle}>
              Name <span style={{ color: theme.accentRed }}>*</span>
            </label>
            <input
              type="text"
              name="Name"
              value={formData.Name}
              onChange={handleChange}
              required
              placeholder="e.g., confidence_weighted_loss"
              style={inputStyle}
            />
          </div>

          {/* Description */}
          <div>
            <label style={labelStyle}>Description</label>
            <textarea
              name="Description"
              value={formData.Description}
              onChange={handleChange}
              placeholder="Describe the research idea: hypothesis, approach, experiment plan, and any relevant context"
              rows={8}
              style={{ ...inputStyle, resize: 'vertical', lineHeight: '1.6' }}
            />
          </div>

          {/* Add to Queue Checkbox */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            padding: '18px 20px',
            background: theme.bgTertiary,
            borderRadius: '8px',
          }}>
            <input
              type="checkbox"
              id="addToQueue"
              checked={addToQueue}
              onChange={(e) => setAddToQueue(e.target.checked)}
              style={{
                width: '18px',
                height: '18px',
                cursor: 'pointer',
                accentColor: theme.accentCoral,
              }}
            />
            <label
              htmlFor="addToQueue"
              style={{
                fontSize: '15px',
                color: theme.textSecondary,
                cursor: 'pointer',
                userSelect: 'none',
              }}
            >
              Add to experiment queue immediately
            </label>
          </div>

          {/* Buttons */}
          <div style={{
            display: 'flex',
            gap: '14px',
            justifyContent: 'flex-end',
            marginTop: '8px',
          }}>
            {onCancel && (
              <button
                type="button"
                onClick={onCancel}
                disabled={submitting}
                style={{
                  padding: '12px 24px',
                  background: theme.bgTertiary,
                  color: theme.textSecondary,
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '15px',
                  fontWeight: '500',
                  cursor: submitting ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s ease',
                }}
              >
                Cancel
              </button>
            )}
            <button
              type="submit"
              disabled={submitting}
              style={{
                padding: '12px 28px',
                background: submitting ? theme.bgElevated : theme.accentCoral,
                color: submitting ? theme.textTertiary : '#FFFFFF',
                border: 'none',
                borderRadius: '8px',
                fontSize: '15px',
                fontWeight: '600',
                cursor: submitting ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s ease',
              }}
            >
              {submitting ? 'Creating...' : 'Create Idea'}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default AddIdeaForm;
