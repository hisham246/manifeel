#!/bin/bash
set -euo pipefail

# Sets up another device (also an RTX 5090, per the earlier conversation -- no base
# image rebuild needed, just the already-built manifeel-isaaclab image) to run training
# via scripts/run_maniflow_usb.sh. Does three things over SSH:
#   1. Clones (or updates) the manifeel repo on the remote -- this is what gets
#      bind-mounted into the container at runtime (-v ...:/workspace/manifeel in
#      run_maniflow_usb.sh), so it must exist standalone on the remote host.
#   2. rsyncs the requested dataset dir(s) under data/ -- these are gitignored
#      (see .gitignore: "data/"), so cloning the repo alone won't bring them.
#   3. Optionally (SYNC_IMAGE=true) streams the manifeel-isaaclab image over via
#      `docker save | ssh ... docker load`, skipping an intermediate 50GB+ tarball
#      on either end.
#
# diffusion_policy/ and maniflow/ do NOT need to exist on the remote at all: they're
# COPYed into the image at build time (see docker/Dockerfile), not bind-mounted, so
# whatever's already baked into the loaded image is all training needs.
#
# Usage:
#   REMOTE=user@host ./scripts/setup_remote.sh
#   REMOTE=user@host DATASETS="usb_quan_Aug05_contact_manifeel usb_quan_Aug05" SYNC_IMAGE=true ./scripts/setup_remote.sh

REMOTE=${REMOTE:?Set REMOTE=user@host (the other device's SSH target)}
REMOTE_DIR=${REMOTE_DIR:-~/hisham/contact_rich_sim}
BRANCH=${BRANCH:-manifeel-isaaclab}
REPO_URL=${REPO_URL:-https://github.com/hisham246/manifeel.git}
DATASETS=${DATASETS:-usb_quan_Aug05_contact_manifeel}
IMAGE=${IMAGE:-manifeel-isaaclab:latest}
SYNC_IMAGE=${SYNC_IMAGE:-false}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== [1/3] Cloning/updating manifeel (branch ${BRANCH}) on ${REMOTE}:${REMOTE_DIR} ==="
ssh "${REMOTE}" "
  set -euo pipefail
  mkdir -p '${REMOTE_DIR}'
  cd '${REMOTE_DIR}'
  if [ -d manifeel/.git ]; then
    cd manifeel && git fetch origin && git checkout '${BRANCH}' && git pull origin '${BRANCH}'
  else
    git clone -b '${BRANCH}' '${REPO_URL}' manifeel
  fi
"

echo "=== [2/3] Syncing dataset(s) [${DATASETS}] ==="
for ds in ${DATASETS}; do
  LOCAL_DS="${REPO_ROOT}/data/${ds}"
  if [ ! -d "${LOCAL_DS}" ]; then
    echo "  SKIP: ${LOCAL_DS} not found locally"
    continue
  fi
  echo "  -> ${ds} ($(du -sh "${LOCAL_DS}" | cut -f1))"
  ssh "${REMOTE}" "mkdir -p '${REMOTE_DIR}/manifeel/data/${ds}'"
  # -P: partial + progress, so a 3-4GB zarr dataset is resumable if the connection drops.
  rsync -azP "${LOCAL_DS}/" "${REMOTE}:${REMOTE_DIR}/manifeel/data/${ds}/"
done

if [ "${SYNC_IMAGE}" = "true" ]; then
  echo "=== [3/3] Streaming docker image ${IMAGE} to ${REMOTE} (this is 50GB+, will take a while) ==="
  docker save "${IMAGE}" | ssh "${REMOTE}" "docker load"
else
  echo "=== [3/3] Skipping image transfer (SYNC_IMAGE=false) ==="
  echo "  If the image isn't already on ${REMOTE}, either re-run with SYNC_IMAGE=true, or manually:"
  echo "    docker save ${IMAGE} | ssh ${REMOTE} docker load"
fi

echo ""
echo "Done. On ${REMOTE}, training can now be launched the same way as here:"
echo "  cd ${REMOTE_DIR}/manifeel && EXTRA_HYDRA_ARGS=\"logging.mode=disabled\" ./scripts/run_maniflow_usb.sh"
