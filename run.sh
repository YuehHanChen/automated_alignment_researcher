#!/bin/bash
# Start the W2S Research web server
#
# Usage:
#   ./run.sh                   # default port 8000
#   ./run.sh 9000              # custom port
#   ./run.sh --no-docker       # skip Docker build
#   ./run.sh --no-docker 9000  # both

set -e

BUILD_DOCKER=true
PORT=8000
for arg in "$@"; do
    case "$arg" in
        --no-docker) BUILD_DOCKER=false ;;
        *) PORT="$arg" ;;
    esac
done
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Load API keys from .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Unset CLAUDECODE to allow nested Claude Code launches from workers
unset CLAUDECODE

# Kill any existing server on this port
pkill -f "python app.py" 2>/dev/null || true
pkill -f "run.py server" 2>/dev/null || true
sleep 1

export GROUND_TRUTH_DIR="$DIR/labeled_data"

# Build Docker image for local Docker mode (if docker is available)
if [ "$BUILD_DOCKER" = true ] && command -v docker &>/dev/null; then
    echo "Building Docker image (w2s-research)..."
    docker build --network host -t w2s-research . && echo "  Done." || echo "  Warning: Docker build failed, Docker mode will not work."
    echo ""
fi

echo "Starting server on port $PORT..."
echo "  Data:         $DIR/data"
echo "  Ground truth: $DIR/labeled_data"
echo "  Cache:        $DIR/cache_results"
echo ""

exec "$DIR/.venv/bin/python" run.py server --port "$PORT"
