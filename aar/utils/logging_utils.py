"""
Shared utilities for conditional Weave tracing.

This module provides utilities to conditionally enable/disable Weave tracing
based on environment variables configured in config.py.
"""
import os
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Check environment variable (can be overridden by config)
# Default: Weave disabled (1) to avoid timeout issues
ENABLE_WEAVE_TRACING = os.getenv("DISABLE_WEAVE_TRACING", "1").lower() not in ("1", "true", "yes")


def set_weave_config(enable_weave: bool):
    """
    Update Weave tracing configuration (typically called from config.py).
    
    Args:
        enable_weave: If True, enable Weave tracing; if False, disable it
    """
    global ENABLE_WEAVE_TRACING
    ENABLE_WEAVE_TRACING = enable_weave


# Weave utilities
_weave_initialized = False
_weave_module = None

if ENABLE_WEAVE_TRACING:
    try:
        import weave
        _weave_available = True
        _weave_module = weave
    except ImportError:
        logger.warning("weave not available, but ENABLE_WEAVE_TRACING is True")
        _weave_available = False
        _weave_module = None
else:
    _weave_available = False
    _weave_module = None


def init_weave(project_name: str = "generalization") -> bool:
    """
    Initialize Weave tracing if enabled.
    
    Args:
        project_name: Project name for Weave
        
    Returns:
        True if Weave was initialized, False otherwise
    """
    global _weave_initialized
    
    if not ENABLE_WEAVE_TRACING:
        logger.info("[Weave] Tracing disabled via DISABLE_WEAVE_TRACING environment variable")
        return False
    
    if _weave_initialized:
        return True
    
    if not _weave_available:
        logger.warning("[Weave] Weave module not available")
        return False
    
    try:
        _weave_module.init(project_name)
        _weave_initialized = True
        logger.info("[Weave] Tracing enabled and initialized")
        return True
    except Exception as e:
        logger.warning(f"[Weave] Failed to initialize Weave tracing: {e}. Continuing without tracing.")
        return False


def get_weave_op():
    """Get weave.op decorator (real or no-op).
    
    Returns a decorator that can be used with or without parentheses:
    - @weave_op
    - @weave_op()
    """
    if ENABLE_WEAVE_TRACING and _weave_available and _weave_initialized:
        return _weave_module.op
    
    # Create a no-op decorator factory that works with or without parentheses
    def noop_weave_op(func=None):
        """No-op decorator that mimics weave.op behavior."""
        if func is None:
            # Called as @weave_op() - return a decorator
            return lambda f: f
        else:
            # Called as @weave_op - return the function unchanged
            return func
    
    return noop_weave_op


def get_weave_attributes():
    """Get weave.attributes context manager (real or no-op)."""
    if ENABLE_WEAVE_TRACING and _weave_available and _weave_initialized:
        return _weave_module.attributes
    
    @contextmanager
    def noop_context(**kwargs):
        yield
    
    return noop_context
