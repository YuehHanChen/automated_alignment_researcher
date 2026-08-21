"""
Shared HTTP utilities for MCP tools.

Provides common HTTP functions used across tool modules:
- Async HTTP POST/GET with httpx fallback to requests
- Server URL resolution from environment variables
- Input validation helpers for security
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

# Try to import httpx for async HTTP, fall back to requests
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    import requests
    HAS_HTTPX = False


# Default headers to bypass Cloudflare bot detection on RunPod proxies
DEFAULT_HEADERS = {
    "User-Agent": "curl/8.0.0",
    "Accept": "*/*",
}


def get_server_url() -> str:
    """
    Get the server URL from environment variable.

    Checks ORCHESTRATOR_API_URL first, then falls back to SERVER_URL.

    Returns:
        Server URL with trailing slash stripped

    Raises:
        ValueError: If neither environment variable is set
    """
    url = os.environ.get("ORCHESTRATOR_API_URL") or os.environ.get("SERVER_URL")
    if not url:
        raise ValueError(
            "ORCHESTRATOR_API_URL or SERVER_URL environment variable is not set. "
            "Cannot access server APIs."
        )
    return url.rstrip("/")


async def async_http_post(
    url: str,
    json_data: Dict[str, Any],
    timeout: int = 60,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Make an async HTTP POST request.

    Uses httpx if available, falls back to synchronous requests.

    Args:
        url: The URL to POST to
        json_data: JSON payload to send
        timeout: Request timeout in seconds
        headers: Optional additional headers

    Returns:
        Parsed JSON response as dict
    """
    h = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
    if headers:
        h.update(headers)

    if HAS_HTTPX:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=json_data, headers=h)
            if response.status_code >= 400:
                try:
                    body = response.json()
                    if "error" in body:
                        return body
                except Exception:
                    pass
                response.raise_for_status()
            return response.json()
    else:
        response = requests.post(url, json=json_data, headers=h, timeout=timeout)
        if response.status_code >= 400:
            try:
                body = response.json()
                if "error" in body:
                    return body
            except Exception:
                pass
            response.raise_for_status()
        return response.json()


async def async_http_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Make an async HTTP GET request.

    Uses httpx if available, falls back to synchronous requests.
    """
    h = {**DEFAULT_HEADERS}
    if headers:
        h.update(headers)

    if HAS_HTTPX:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params=params, headers=h)
            response.raise_for_status()
            return response.json()
    else:
        response = requests.get(url, params=params, headers=h, timeout=timeout)
        response.raise_for_status()
        return response.json()


def validate_safe_identifier(value: str, name: str) -> str:
    """
    Validate that an identifier is safe (no path traversal).

    Args:
        value: The identifier to validate
        name: Name of the parameter (for error messages)

    Returns:
        The validated and stripped identifier

    Raises:
        ValueError: If validation fails
    """
    if not value:
        raise ValueError(f"{name} is required")

    value = value.strip()

    if ".." in value or "/" in value or "\\" in value:
        raise ValueError(f"{name} contains invalid characters (path traversal attempt)")

    if not re.match(r'^[a-zA-Z0-9_\-]+$', value):
        raise ValueError(
            f"{name} contains invalid characters (only alphanumeric, underscore, hyphen allowed)"
        )

    return value


def validate_safe_path(path: str, name: str, allowed_base: str = "/workspace") -> str:
    """
    Validate that a path is safe and within allowed base directory.

    Prevents path traversal attacks by resolving to absolute path
    and checking containment.

    Args:
        path: The path to validate
        name: Name of the parameter (for error messages)
        allowed_base: Base directory that path must be within

    Returns:
        The resolved absolute path as string

    Raises:
        ValueError: If validation fails
    """
    if not path:
        raise ValueError(f"{name} is required")

    path = path.strip()

    resolved = Path(path).resolve()
    allowed_resolved = Path(allowed_base).resolve()

    try:
        resolved.relative_to(allowed_resolved)
    except ValueError:
        raise ValueError(
            f"{name} must be within {allowed_base}. "
            f"Got: {resolved} (path traversal attempt)"
        )

    return str(resolved)
