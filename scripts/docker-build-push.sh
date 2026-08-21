#!/bin/bash

# Script to build Docker image on Mac and push to Docker Hub
# This script does NOT require GPU (Mac-friendly)
#
# Usage:
#   ./docker-build-push.sh [tag]
#   ./docker-build-push.sh dev
#   ./docker-build-push.sh v1.0.0
#   IMAGE_TAG=dev ./docker-build-push.sh
#
# If no tag is provided, defaults to "latest"

set -e  # Exit on error

echo "=========================================="
echo "W2S Research Docker Build & Push (Mac)"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse command-line arguments
# Priority: command-line arg > environment variable > default
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [tag]"
    echo ""
    echo "Build and push Docker image to Docker Hub"
    echo ""
    echo "Arguments:"
    echo "  tag    Docker image tag (default: latest)"
    echo ""
    echo "Examples:"
    echo "  $0 dev                    # Build and push with tag 'dev'"
    echo "  $0 v1.0.0                 # Build and push with tag 'v1.0.0'"
    echo "  IMAGE_TAG=dev $0          # Use environment variable"
    echo "  $0                        # Use default tag 'latest'"
    echo ""
    echo "Environment variables:"
    echo "  DOCKER_USERNAME           Docker Hub username (required)"
    echo "  IMAGE_TAG                 Image tag (optional, can be overridden by argument)"
    exit 0
fi

if [ $# -gt 0 ]; then
    IMAGE_TAG="$1"
    echo -e "${BLUE}Using tag from command line: ${IMAGE_TAG}${NC}"
elif [ -n "${IMAGE_TAG}" ]; then
    echo -e "${BLUE}Using tag from environment: ${IMAGE_TAG}${NC}"
else
    IMAGE_TAG="latest"
    echo -e "${YELLOW}No tag specified, using default: ${IMAGE_TAG}${NC}"
fi

# Configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
IMAGE_NAME="w2s-research"

# Step 1: Check prerequisites
echo -e "${BLUE}Step 1: Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

if [ -z "$DOCKER_USERNAME" ]; then
    echo -e "${YELLOW}DOCKER_USERNAME not set. Please provide your Docker Hub username:${NC}"
    read -p "Docker Hub Username: " DOCKER_USERNAME
    if [ -z "$DOCKER_USERNAME" ]; then
        echo -e "${RED}Error: Docker Hub username is required${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Docker installed${NC}"
echo -e "${GREEN}✓ Docker Hub username: ${DOCKER_USERNAME}${NC}"
echo ""

# Step 2: Login to Docker Hub
echo -e "${BLUE}Step 2: Logging in to Docker Hub...${NC}"
if ! docker login; then
    echo -e "${RED}Docker Hub login failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Hub login successful${NC}"
echo ""

# Step 3: Build Docker image
echo -e "${BLUE}Step 3: Building Docker image...${NC}"
echo "This may take 15-30 minutes depending on your internet connection"
echo "Building: ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# Build with platform specification for compatibility
docker build \
    --platform linux/amd64 \
    -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG} \
    -t ${DOCKER_USERNAME}/${IMAGE_NAME}:$(date +%Y%m%d) \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${RED}Docker build failed${NC}"
    exit 1
fi
echo ""

# Step 4: Check image size
echo -e "${BLUE}Step 4: Checking image size...${NC}"
docker images ${DOCKER_USERNAME}/${IMAGE_NAME}
echo ""

# Step 5: Push to Docker Hub
echo -e "${BLUE}Step 6: Pushing to Docker Hub...${NC}"
echo "This may take 5-15 minutes depending on your upload speed"
echo ""

docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:$(date +%Y%m%d)

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Image pushed to Docker Hub successfully${NC}"
else
    echo -e "${RED}Docker push failed${NC}"
    exit 1
fi
echo ""

# Success summary
DATE_TAG=$(date +%Y%m%d)
echo -e "${GREEN}=========================================="
echo "Build and Push Complete! ✓"
echo "==========================================${NC}"
echo ""
echo "Images pushed to Docker Hub:"
echo "  • ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG} (primary tag)"
echo "  • ${DOCKER_USERNAME}/${IMAGE_NAME}:${DATE_TAG} (date tag for backup)"
echo ""
echo "Next steps:"
echo "  1. Create a RunPod template using: ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "  2. Set RUNPOD_TEMPLATE_ID environment variable to your new template ID"
echo "  3. Restart your Flask server to use the new template"
echo ""
echo "Or test locally:"
echo "  docker run --rm --gpus all \\"
echo "    -v \$(pwd)/data:/workspace/data \\"
echo "    -v \$(pwd)/results:/workspace/results \\"
echo "    ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG} \\"
echo "    python run.py --idea vanilla_w2s --seed 42"
echo ""
