#!/bin/bash
# Publish the AAR dashboard as a static site to SiteGround. RUN FROM YOUR LAPTOP
# (it bridges the cluster — over SSH — and SiteGround — over SSH/rsync or SFTP).
#
# Flow: ssh→cluster regenerates a self-contained static snapshot → rsync it down →
# rsync/lftp it up to SiteGround's doc root. Put this on a cron/launchd every few
# minutes for "live" updates.
#
# One-time setup — set these (e.g. in ~/.aar_site_env, then `source` it):
#   export SG_HOST=...          # SiteGround SSH/SFTP host (Site Tools → Security → SSH Keys / FTP Accounts)
#   export SG_USER=...          # SiteGround SSH/SFTP username
#   export SG_PORT=18765        # SiteGround SFTP/SSH port (default 18765)
#   export SG_DOCROOT=public_html/aar_dashboard # → served at https://www.john-chen.cc/aar_dashboard/
#   # (root of the site would be just: SG_DOCROOT=public_html)
#   export SG_AUTH=ssh          # "ssh" (rsync over SSH key — recommended, needs SSH enabled + key added)
#                               # or "sftp" (password via lftp; needs `brew install lftp`)
#   export SG_PASS=...          # only for SG_AUTH=sftp
# Usage:  source ~/.aar_site_env && scripts/publish_site.sh
set -euo pipefail

CLUSTER_SSH="${CLUSTER_SSH:-cluster}"                 # your ~/.ssh/config alias for the cluster
REMOTE_REPO="${REMOTE_REPO:-/opt/aar/aar_repo}"
REMOTE_VENV="${REMOTE_VENV:-/opt/aar/work/dashboard_venv/bin/python}"
REMOTE_EXPORT="${REMOTE_EXPORT:-/opt/aar/work/aar_site_export}"
LOCAL_STAGE="${LOCAL_STAGE:-/tmp/aar_site}"
: "${SG_HOST:?set SG_HOST}"; : "${SG_USER:?set SG_USER}"
SG_PORT="${SG_PORT:-18765}"; SG_DOCROOT="${SG_DOCROOT:-public_html/aar_dashboard}"; SG_AUTH="${SG_AUTH:-ssh}"

echo "[publish] 1/3 regenerating static snapshot on the cluster…"
ssh "${CLUSTER_SSH}" \
  "PYTHONPATH='${REMOTE_REPO}' FORUM_ROOT=/opt/aar/work \
   '${REMOTE_VENV}' -m aar.web_ui.live_dashboard --static-export '${REMOTE_EXPORT}'"

echo "[publish] 2/3 pulling snapshot to ${LOCAL_STAGE}…"
rm -rf "${LOCAL_STAGE}"; mkdir -p "${LOCAL_STAGE}"
rsync -az --delete "${CLUSTER_SSH}:${REMOTE_EXPORT}/" "${LOCAL_STAGE}/"

echo "[publish] 3/3 uploading to SiteGround (${SG_AUTH}) → ${SG_DOCROOT}…"
if [ "${SG_AUTH}" = "ssh" ]; then
  # rsync over SSH (recommended; add your laptop's SSH key in Site Tools → Security → SSH Keys)
  rsync -az --delete -e "ssh -p ${SG_PORT}" "${LOCAL_STAGE}/" "${SG_USER}@${SG_HOST}:${SG_DOCROOT}/"
else
  # SFTP via lftp with a password (brew install lftp)
  lftp -u "${SG_USER},${SG_PASS:?set SG_PASS for sftp}" -p "${SG_PORT}" "sftp://${SG_HOST}" \
       -e "set sftp:auto-confirm yes; mirror -R --delete --parallel=4 '${LOCAL_STAGE}/' '${SG_DOCROOT}/'; bye"
fi
echo "[publish] done → https://www.john-chen.cc/"
