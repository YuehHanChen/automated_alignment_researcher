"""
Runpod deployment utilities for deploying pods and executing commands.

This module provides functions to deploy Runpod pods with configurable commands
and environment variables.
"""

import os
import requests
import shlex
import sys
from typing import Dict, List, Optional, Any


class RunPodCapacityError(Exception):
    """Raised when RunPod has insufficient capacity (transient, retryable)."""
    pass


class RunPodPermanentError(Exception):
    """Raised for permanent errors (bad API key, invalid template)."""
    pass


# Runpod API URL
RUNPOD_API_URL = "https://rest.runpod.io/v1/pods"

# Default datacenter IDs for secure cloud deployments (All regions)
DEFAULT_DATACENTER_IDS = [
    # North America
    "CA-MTL-1",   # Canada - Montreal
    "CA-MTL-2",   # Canada - Montreal
    "CA-MTL-3",   # Canada - Montreal
    "US-CA-2",    # US - California
    "US-DE-1",    # US - Delaware
    "US-GA-1",    # US - Georgia
    "US-GA-2",    # US - Georgia
    "US-IL-1",    # US - Illinois
    "US-KS-2",    # US - Kansas
    "US-KS-3",    # US - Kansas
    "US-NC-1",    # US - North Carolina
    "US-TX-1",    # US - Texas
    "US-TX-3",    # US - Texas
    "US-TX-4",    # US - Texas
    "US-WA-1",    # US - Washington
    # Europe
    "EU-RO-1",    # Europe - Romania
    "EU-SE-1",    # Europe - Sweden
    "EU-CZ-1",    # Europe - Czech Republic
    "EU-NL-1",    # Europe - Netherlands
    "EU-FR-1",    # Europe - France
    "EUR-IS-1",   # Europe - Iceland
    "EUR-IS-2",   # Europe - Iceland
    "EUR-IS-3",   # Europe - Iceland
    "EUR-NO-1",   # Europe - Norway
    # Asia Pacific
    "AP-JP-1",    # Asia Pacific - Japan
    # Oceania
    "OC-AU-1",    # Oceania - Australia
]


def create_run_command(
    command: List[str],
    env_vars: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Create a bash command string with environment variables exported.
    
    Args:
        command: List of command parts (e.g., ["python", "-m", "aar.main"])
        env_vars: Dictionary of environment variables to export
        
    Returns:
        List with ["bash", "-c", "<command_string>"] format
    """
    if env_vars is None:
        env_vars = {}
    
    # Build export statements for environment variables
    export_parts = []
    for key, value in env_vars.items():
        escaped_value = shlex.quote(value)
        export_parts.append(f"export {key}={escaped_value}")
    
    # Build the full command
    # Escape each command part and join with spaces
    escaped_command_parts = [shlex.quote(part) for part in command]
    command_str = " ".join(escaped_command_parts)
    
    # Combine: export vars && cd to workspace && run command
    # Note: Workspace path should match what's in the RunPod template
    from aar.config import WORKSPACE_DIR as default_ws
    workspace_dir = os.getenv("WORKSPACE_DIR", default_ws)
    full_command = " && ".join([
        *export_parts,
        f"cd {workspace_dir}",
        command_str,
    ])
    
    return ["bash", "-c", full_command]


def deploy_pod(
    command: List[str],
    env_vars: Optional[Dict[str, str]] = None,
    pod_name: str = "aar",
    template_id: str = "",
    gpu_count: int = 1,
    gpu_type_ids: Optional[List[str]] = None,
    cloud_type: str = "SECURE",
    global_networking: bool = True,
    compute_type: str = "GPU",
    gpu_type_priority: str = "availability",
    data_center_ids: Optional[List[str]] = None,
    data_center_priority: str = "availability",
    interruptible: bool = False,
    locked: bool = False,
    runpod_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deploy a Runpod pod with the specified command and configuration.
    
    Args:
        command: List of command parts to execute (e.g., ["python", "-m", "aar.main"])
        env_vars: Dictionary of environment variables to export
        pod_name: Name for the pod
        template_id: Runpod template ID
        gpu_count: Number of GPUs
        gpu_type_ids: List of GPU type IDs (default: ["NVIDIA H200"])
        cloud_type: Cloud type (default: "SECURE")
        global_networking: Enable global networking (default: True)
        compute_type: Compute type (default: "GPU")
        gpu_type_priority: GPU type priority (default: "availability")
        data_center_ids: List of datacenter IDs to deploy to (default: DEFAULT_DATACENTER_IDS)
        data_center_priority: Datacenter priority mode (default: "availability")
        interruptible: Whether pod is interruptible (default: False)
        locked: Whether pod is locked (default: False)
        runpod_api_key: Runpod API key (if None, will try env var or prompt)
        
    Returns:
        Response dictionary from Runpod API
    """
    if gpu_type_ids is None:
        gpu_type_ids = ["NVIDIA H200"]
    
    if data_center_ids is None:
        data_center_ids = DEFAULT_DATACENTER_IDS
    
    # Get Runpod API key
    if runpod_api_key is None:
        runpod_api_key = os.getenv("RUNPOD_API_KEY")
        if not runpod_api_key:
            print("RUNPOD_API_KEY not found in environment variables.")
            runpod_api_key = input("Please enter your RUNPOD_API_KEY: ").strip()
            if not runpod_api_key:
                print("Error: RUNPOD_API_KEY is required.")
                sys.exit(1)
    
    # Create the run command with environment variables
    docker_start_cmd = create_run_command(command, env_vars)
    
    # Build payload
    payload = {
        "name": pod_name,
        "templateId": template_id,
        "gpuCount": gpu_count,
        "gpuTypeIds": gpu_type_ids,
        "cloudType": cloud_type,
        "globalNetworking": global_networking,
        "computeType": compute_type,
        "gpuTypePriority": gpu_type_priority,
        "dataCenterIds": data_center_ids,
        "dataCenterPriority": data_center_priority,
        "interruptible": interruptible,
        "locked": locked,
        "dockerStartCmd": docker_start_cmd,
    }
    
    # Set headers
    headers = {
        "Authorization": f"Bearer {runpod_api_key}",
        "Content-Type": "application/json",
    }
    
    # Make API request
    response = requests.post(RUNPOD_API_URL, json=payload, headers=headers)

    if not response.ok:
        error_body = response.text
        status_code = response.status_code

        # Transient errors - capacity/availability issues (retryable)
        transient_keywords = ["capacity", "unavailable", "no available", "no instances", "insufficient", "try again", "gpu"]
        if status_code in (429, 503, 507) or any(kw in error_body.lower() for kw in transient_keywords):
            raise RunPodCapacityError(f"RunPod capacity unavailable (HTTP {status_code}): {error_body}")

        # Permanent errors - fail immediately
        raise RunPodPermanentError(f"RunPod deployment failed (HTTP {status_code}): {error_body}")

    return response.json()


def get_pod_status(pod_id: str, runpod_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Get status of a RunPod pod.
    
    Args:
        pod_id: RunPod pod ID
        runpod_api_key: RunPod API key (if None, will try env var)
        
    Returns:
        Pod status dictionary
    """
    if runpod_api_key is None:
        runpod_api_key = os.getenv("RUNPOD_API_KEY")
        if not runpod_api_key:
            raise ValueError("RUNPOD_API_KEY not found")
    
    headers = {
        "Authorization": f"Bearer {runpod_api_key}",
        "Content-Type": "application/json",
    }
    
    response = requests.get(f"{RUNPOD_API_URL}/{pod_id}", headers=headers)
    response.raise_for_status()
    
    return response.json()


def stop_pod(pod_id: str, runpod_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Stop a RunPod pod.
    
    Args:
        pod_id: RunPod pod ID
        runpod_api_key: RunPod API key (if None, will try env var)
        
    Returns:
        Response dictionary from RunPod API
    """
    if runpod_api_key is None:
        runpod_api_key = os.getenv("RUNPOD_API_KEY")
        if not runpod_api_key:
            raise ValueError("RUNPOD_API_KEY not found")
    
    headers = {
        "Authorization": f"Bearer {runpod_api_key}",
        "Content-Type": "application/json",
    }
    
    response = requests.post(f"{RUNPOD_API_URL}/{pod_id}/stop", headers=headers)
    response.raise_for_status()
    
    return response.json()


def delete_pod(pod_id: str, runpod_api_key: Optional[str] = None) -> None:
    """
    Delete (terminate) a RunPod pod permanently.
    
    This permanently deletes the pod and all associated data.
    Use with caution - this action cannot be undone.
    
    Args:
        pod_id: RunPod pod ID
        runpod_api_key: RunPod API key (if None, will try env var)
        
    Returns:
        None (204 No Content on success)
    """
    if runpod_api_key is None:
        runpod_api_key = os.getenv("RUNPOD_API_KEY")
        if not runpod_api_key:
            raise ValueError("RUNPOD_API_KEY not found")
    
    headers = {
        "Authorization": f"Bearer {runpod_api_key}",
        "Content-Type": "application/json",
    }
    
    response = requests.delete(f"{RUNPOD_API_URL}/{pod_id}", headers=headers)
    response.raise_for_status()
    
    # DELETE endpoint returns 204 No Content on success
    return None


def get_api_key_from_env_or_prompt(key_name: str, description: str = "") -> str:
    """
    Get API key from environment variable or prompt user.
    
    Args:
        key_name: Name of the environment variable
        description: Optional description to show when prompting
        
    Returns:
        API key string
    """
    api_key = os.getenv(key_name)
    if not api_key:
        prompt = f"{key_name} not found in environment variables."
        if description:
            prompt += f" {description}"
        print(prompt)
        api_key = input(f"Please enter your {key_name}: ").strip()
        if not api_key:
            print(f"Error: {key_name} is required.")
            sys.exit(1)
    return api_key



