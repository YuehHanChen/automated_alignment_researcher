#!/bin/bash

# Generate SSH host keys (must be done before starting SSH service)
if command -v ssh-keygen >/dev/null 2>&1; then
    mkdir -p /etc/ssh
    ssh-keygen -A
fi

# Add public keys to authorized_keys (if PUBLIC_KEY env var is set by Runpod)
# Add to both root and ubuntu-cmd user
if [ -n "$PUBLIC_KEY" ]; then
    # Root user
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    
    # ubuntu-cmd user
    mkdir -p /home/ubuntu-cmd/.ssh
    chmod 700 /home/ubuntu-cmd/.ssh
    echo "$PUBLIC_KEY" > /home/ubuntu-cmd/.ssh/authorized_keys
    chmod 600 /home/ubuntu-cmd/.ssh/authorized_keys
    chown -R ubuntu-cmd:ubuntu-cmd /home/ubuntu-cmd/.ssh
fi

# Start SSH service
if command -v sshd >/dev/null 2>&1; then
    service ssh start || /usr/sbin/sshd
fi

# Grant ubuntu-cmd full access to /workspace (simpler and more robust)
# This ensures all subdirectories (including .cache/huggingface) have correct permissions
echo "Granting ubuntu-cmd full access to /workspace..."
chown -R ubuntu-cmd:ubuntu-cmd /workspace 2>/dev/null || true
chmod -R 755 /workspace 2>/dev/null || true
echo "✓ ubuntu-cmd now has full access to /workspace"

# Copy code to /workspace/automated-w2s-research (Runpod mounts /workspace after container starts)
# Only copy if /workspace/automated-w2s-research doesn't exist or is empty (avoid overwriting user changes)
if [ ! -d "/workspace/automated-w2s-research" ] || [ -z "$(ls -A /workspace/automated-w2s-research 2>/dev/null)" ]; then
    echo "Copying code to /workspace/automated-w2s-research..."
    mkdir -p /workspace/automated-w2s-research
    # Use /opt/automated-w2s-research/. to copy ALL files including hidden ones (like .claude/)
    cp -r /opt/automated-w2s-research/. /workspace/automated-w2s-research/ 2>/dev/null || true
    # Make workspace accessible to ubuntu-cmd user
    chown -R ubuntu-cmd:ubuntu-cmd /workspace/automated-w2s-research 2>/dev/null || true
    echo "Code copied to /workspace/automated-w2s-research"
fi

# Always pull latest code from git if GIT_PULL_ON_START is set
# This allows using older images with newer code
if [ "${GIT_PULL_ON_START:-false}" = "true" ] && [ -d "/workspace/automated-w2s-research/.git" ]; then
    echo "Pulling latest code from git..."
    cd /workspace/automated-w2s-research
    git fetch origin
    git reset --hard origin/${GIT_BRANCH:-dontstop}
    chown -R ubuntu-cmd:ubuntu-cmd /workspace/automated-w2s-research 2>/dev/null || true
    echo "✓ Code updated to latest ${GIT_BRANCH:-dontstop}"
fi

# Read and export RUNPOD_POD_ID if available (RunPod sets this automatically)
# This allows the pod to delete itself when experiments complete
if [ -n "$RUNPOD_POD_ID" ]; then
    export RUNPOD_POD_ID="$RUNPOD_POD_ID"
    echo "✓ RUNPOD_POD_ID set: ${RUNPOD_POD_ID}"
else
    echo "⚠️  RUNPOD_POD_ID not found in environment"
    echo "   Note: RunPod should set this automatically. If not, orchestrator will handle cleanup."
fi

# Always switch to ubuntu-cmd user (non-root) for security
# Claude Code CLI should NOT run as root for security reasons
# Execute the command (default: sleep infinity, or overridden by dockerStartCmd) as ubuntu-cmd
# Note: Environment variables should be exported in run_command using 'export' in bash -c
# (e.g., ["bash", "-c", "export ANTHROPIC_API_KEY=... && uv run python ..."])

# Build command string with all arguments properly quoted
# $@ will contain CMD from Dockerfile (default: ["sleep", "infinity"])
# or be overridden by dockerStartCmd from deployment script (e.g., aar.infrastructure.runpod)
CMD="cd /workspace/automated-w2s-research &&"
for arg in "$@"; do
    # Escape single quotes: replace ' with '\''
    escaped_arg=$(printf '%s\n' "$arg" | sed "s/'/'\"'\"'/g")
    CMD="${CMD} '${escaped_arg}'"
done

# Execute as ubuntu-cmd user, preserving RUNPOD_POD_ID environment variable
# Use 'su' (not 'su -') to preserve environment variables
if [ -n "$RUNPOD_POD_ID" ]; then
    exec su ubuntu-cmd -c "export RUNPOD_POD_ID=\"$RUNPOD_POD_ID\" && $CMD"
else
    exec su ubuntu-cmd -c "$CMD"
fi
