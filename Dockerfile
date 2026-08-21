# Use Runpod's official PyTorch base image
# CUDA 12.8.1, PyTorch 2.8.0, Ubuntu 24.04
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Install system dependencies first
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        openssh-server \
        git \
        curl \
    && mkdir -p /var/run/sshd && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager) to system location
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    chmod +x /usr/local/bin/uv

# Create non-root user (no sudo access required - Claude Code CLI with bypassPermissions mode
# requires non-root user without sudo privileges for security)
RUN useradd -m -s /bin/bash ubuntu-cmd && \
    mkdir -p /home/ubuntu-cmd/.ssh && \
    chmod 700 /home/ubuntu-cmd/.ssh

# Install Claude Code CLI as ubuntu-cmd user (recommended method)
# Reference: https://code.claude.com/docs/en/setup
# Installing as ubuntu-cmd ensures marketplace is properly initialized for that user
RUN su ubuntu-cmd -c "curl -fsSL https://claude.ai/install.sh | bash" && \
    # Verify installation
    su ubuntu-cmd -c "/home/ubuntu-cmd/.local/bin/claude --version" && \
    # Make claude available in PATH by creating symlink (pointing to ubuntu-cmd's installation)
    ln -s /home/ubuntu-cmd/.local/bin/claude /usr/local/bin/claude

# Update PATH to include system-wide binaries (needed for ubuntu-cmd user)
ENV PATH="/usr/local/bin:$PATH"

# Install ralph-loop plugin for infinite loop iteration
# This plugin enables the /ralph-loop:ralph-loop command used in autonomous baseline
# Note: Marketplace is not auto-configured in Docker build, so we add it explicitly
RUN echo "=== Claude Code version ===" && \
    su ubuntu-cmd -c "claude --version" && \
    echo "=== Adding official marketplace ===" && \
    su ubuntu-cmd -c "claude plugin marketplace add https://github.com/anthropics/claude-plugins-official" && \
    echo "=== Available marketplaces ===" && \
    su ubuntu-cmd -c "claude plugin marketplace list" && \
    echo "=== Installing ralph-loop plugin ===" && \
    su ubuntu-cmd -c "claude plugin install ralph-loop" && \
    echo "=== Installed plugins ===" && \
    su ubuntu-cmd -c "claude plugin list" && \
    echo "✓ ralph-loop plugin installed for ubuntu-cmd user"

# Copy project files for better caching
# Build code in /opt/automated-w2s-research (will be copied to /workspace/automated-w2s-research at runtime)
WORKDIR /opt/automated-w2s-research
COPY pyproject.toml uv.lock CLAUDE.md README.md ./

# Install Python dependencies from lock file (system-wide for Docker)
# Export lock file to requirements format and install system-wide
# Note: --break-system-packages is needed to bypass PEP 668 protection in Ubuntu 24.04
# Note: --no-cache prevents disk space exhaustion from large packages like sgl-kernel
RUN uv export --all-groups --format requirements-txt --no-hashes -o /tmp/requirements.txt && \
    uv pip install --system --break-system-packages --no-cache -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Configure SSH (allow both root and ubuntu-cmd)
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "AllowUsers root ubuntu-cmd" >> /etc/ssh/sshd_config

# Copy the entire project to /opt/automated-w2s-research (temporary build location, copied to /workspace at runtime)
COPY . .

# Install the project (and verl if present) in editable mode (system-wide)
RUN uv pip install -e . --system --break-system-packages --no-cache && \
    if [ -f verl/pyproject.toml ] || [ -f verl/setup.py ]; then \
        uv pip install --no-deps -e verl --system --break-system-packages --no-cache; \
    fi

# Grant ubuntu-cmd write access to workspace (needed for local Docker mode)
RUN chown -R ubuntu-cmd:ubuntu-cmd /opt/automated-w2s-research

# Copy and set up entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/opt/automated-w2s-research
ENV VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Expose SSH port
EXPOSE 22
EXPOSE 8000

# Set entrypoint and default command
# Default behavior: switch to ubuntu-cmd user and sleep (for interactive use)
# Override via dockerStartCmd in deployment script (e.g., aar.infrastructure.runpod)
ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
