#!/usr/bin/env bash
# Setup script for self-hosted GitHub Actions runner with NVIDIA GPU
# Run with: sudo bash .github/setup-runner.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo)."
    exit 1
fi

RUNNER_USER="github-runner"
RUNNER_HOME="/home/${RUNNER_USER}"
RUNNER_DIR="${RUNNER_HOME}/actions-runner"
RUNNER_VERSION="2.322.0"

echo "============================================================"
echo "GitHub Actions Self-Hosted Runner Setup (GPU)"
echo "============================================================"

# --- System packages ---
echo ""
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq curl tar build-essential cmake git jq

# --- Create unprivileged runner user ---
echo ""
echo "[2/7] Creating runner user '${RUNNER_USER}'..."
if id "${RUNNER_USER}" &>/dev/null; then
    echo "  User '${RUNNER_USER}' already exists, skipping."
else
    useradd -m -s /bin/bash "${RUNNER_USER}"
    echo "  User '${RUNNER_USER}' created."
fi

# --- NVIDIA driver ---
echo ""
echo "[3/7] Installing NVIDIA drivers..."
if command -v nvidia-smi &>/dev/null; then
    echo "  NVIDIA driver already installed:"
    nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
else
    apt-get install -y -qq ubuntu-drivers-common
    ubuntu-drivers autoinstall
    echo ""
    echo "  *** NVIDIA driver installed. A REBOOT is required. ***"
    echo "  Re-run this script after rebooting to complete setup."
    exit 0
fi

# --- Add runner user to GPU groups ---
echo ""
echo "[4/7] Adding '${RUNNER_USER}' to video and render groups..."
usermod -aG video "${RUNNER_USER}"
usermod -aG render "${RUNNER_USER}"

# --- Create hostedtoolcache for julia-actions/setup-julia ---
echo ""
echo "[5/7] Creating /opt/hostedtoolcache..."
mkdir -p /opt/hostedtoolcache
chown "${RUNNER_USER}:${RUNNER_USER}" /opt/hostedtoolcache

# --- Download GitHub Actions runner ---
echo ""
echo "[6/7] Downloading GitHub Actions runner v${RUNNER_VERSION}..."
mkdir -p "${RUNNER_DIR}"
cd "${RUNNER_DIR}"

RUNNER_ARCH="x64"
RUNNER_TAR="actions-runner-linux-${RUNNER_ARCH}-${RUNNER_VERSION}.tar.gz"
if [[ ! -f "${RUNNER_DIR}/run.sh" ]]; then
    curl -sL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_TAR}" -o "${RUNNER_TAR}"
    tar xzf "${RUNNER_TAR}"
    rm -f "${RUNNER_TAR}"
    chown -R "${RUNNER_USER}:${RUNNER_USER}" "${RUNNER_DIR}"
    echo "  Runner extracted to ${RUNNER_DIR}"
else
    echo "  Runner already installed, skipping download."
fi

# --- Cron jobs ---
echo ""
echo "[7/7] Setting up cron jobs..."

# Weekly Julia depot cleanup (Sunday 3am)
DEPOT_CRON="0 3 * * 0 find ${RUNNER_HOME}/.julia/compiled -type f -mtime +30 -delete 2>/dev/null; find ${RUNNER_HOME}/.julia/packages -maxdepth 2 -type d -empty -delete 2>/dev/null"

# Hourly GPU health check
GPU_HEALTH_CRON="0 * * * * nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader >> /var/log/gpu-health.log 2>&1"

(crontab -u "${RUNNER_USER}" -l 2>/dev/null || true; echo "${DEPOT_CRON}"; echo "${GPU_HEALTH_CRON}") | sort -u | crontab -u "${RUNNER_USER}" -
echo "  Cron jobs installed for '${RUNNER_USER}'."

# --- Done ---
echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Register the runner with your repository:"
echo "     su - ${RUNNER_USER}"
echo "     cd ${RUNNER_DIR}"
echo "     ./config.sh --url https://github.com/OWNER/REPO \\"
echo "                 --token YOUR_REGISTRATION_TOKEN \\"
echo "                 --labels self-hosted,linux,x64,gpu \\"
echo "                 --name \$(hostname) \\"
echo "                 --work _work"
echo ""
echo "  2. Install and start the systemd service:"
echo "     cd ${RUNNER_DIR}"
echo "     sudo ./svc.sh install ${RUNNER_USER}"
echo "     sudo ./svc.sh start"
echo ""
echo "  3. Verify the runner appears in your repo's Settings > Actions > Runners"
echo ""
