#!/usr/bin/env bash
# Setup script for self-hosted GitHub Actions runner (CPU)
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
echo "GitHub Actions Self-Hosted Runner Setup"
echo "============================================================"

# --- System packages ---
echo ""
echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y -qq curl tar build-essential cmake git jq

# --- Create unprivileged runner user ---
echo ""
echo "[2/5] Creating runner user '${RUNNER_USER}'..."
if id "${RUNNER_USER}" &>/dev/null; then
    echo "  User '${RUNNER_USER}' already exists, skipping."
else
    useradd -m -s /bin/bash "${RUNNER_USER}"
    echo "  User '${RUNNER_USER}' created."
fi

# --- Create hostedtoolcache for julia-actions/setup-julia ---
echo ""
echo "[3/5] Creating /opt/hostedtoolcache..."
mkdir -p /opt/hostedtoolcache
chown "${RUNNER_USER}:${RUNNER_USER}" /opt/hostedtoolcache

# --- Download GitHub Actions runner ---
echo ""
echo "[4/5] Downloading GitHub Actions runner v${RUNNER_VERSION}..."
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
echo "[5/5] Setting up cron jobs..."

# Weekly Julia depot cleanup (Sunday 3am)
DEPOT_CRON="0 3 * * 0 find ${RUNNER_HOME}/.julia/compiled -type f -mtime +30 -delete 2>/dev/null; find ${RUNNER_HOME}/.julia/packages -maxdepth 2 -type d -empty -delete 2>/dev/null"

(crontab -u "${RUNNER_USER}" -l 2>/dev/null || true; echo "${DEPOT_CRON}") | sort -u | crontab -u "${RUNNER_USER}" -
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
echo "                 --labels self-hosted,linux,x64 \\"
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
