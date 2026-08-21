"""
MCP tools for the research loop.

- prior_work_tools: Snapshot download
- server_api_tools: Server API tools (evaluate, share, leaderboard)
- http_utils: Shared HTTP utilities used by all tool modules

Snapshots are automatically created via share_finding (finding_type='result').
"""

# Shared HTTP utilities
from .http_utils import (
    get_server_url,
    async_http_post,
    async_http_get,
    validate_safe_identifier,
    validate_safe_path,
    HAS_HTTPX,
    DEFAULT_HEADERS,
)

# Prior work tools (snapshots)
try:
    from .prior_work_tools import (
        create_prior_work_tools_server,
        download_snapshot,
    )
except ImportError:
    download_snapshot = None
    create_prior_work_tools_server = None

# Server API tools
try:
    from .server_api_tools import (
        create_server_api_tools_server,
        evaluate_predictions,
        share_finding,
        get_leaderboard,
    )
except ImportError:
    evaluate_predictions = None
    share_finding = None
    get_leaderboard = None
    create_server_api_tools_server = None


__all__ = [
    # HTTP utilities
    "get_server_url",
    "async_http_post",
    "async_http_get",
    "validate_safe_identifier",
    "validate_safe_path",
    "HAS_HTTPX",
    # Prior work tools
    "create_prior_work_tools_server",
    "download_snapshot",
    # Server API tools
    "create_server_api_tools_server",
    "evaluate_predictions",
    "share_finding",
    "get_leaderboard",
]
