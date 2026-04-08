#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/ttt-e2e-data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/author_checkpoints}"
PAPER_RUN_ID="${PAPER_RUN_ID:-protocol_r_760m_author_seed_v1}"
EXP_FOLDER="${EXP_FOLDER:-protocol_r_760m_author_seed_v1}"
PHASE="${PHASE:-core}"
SAVE_MILESTONE_FREQ="${SAVE_MILESTONE_FREQ:-2400}"
ATTN_IMPL="${ATTN_IMPL:-none}"
SWA_PREFIX_ATTN_IMPL="${SWA_PREFIX_ATTN_IMPL:-}"
B2_ARTIFACT_PREFIX="${B2_ARTIFACT_PREFIX:-ttt-e2e-artifacts}"
B2_DATASET_PACKAGE="${B2_DATASET_PACKAGE:-paper_budget_760m_val-full}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-600}"
RUN_LOG="${RUN_LOG:-/workspace/spot_resume_ladder.log}"
SYNC_LOG="${SYNC_LOG:-/workspace/spot_resume_b2_sync.log}"
AWS_RETRY_MODE="${AWS_RETRY_MODE:-adaptive}"
AWS_MAX_ATTEMPTS="${AWS_MAX_ATTEMPTS:-12}"
SYNC_PID=""

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required env var: $name" >&2
    exit 1
  fi
}

require_env AWS_ACCESS_KEY_ID
require_env AWS_SECRET_ACCESS_KEY
require_env AWS_DEFAULT_REGION
require_env B2_ENDPOINT_URL
require_env B2_BUCKET
require_env B2_DATASET_PREFIX

mkdir -p "$DATA_ROOT" "$CHECKPOINT_ROOT" "$ARTIFACT_ROOT"
export AWS_RETRY_MODE AWS_MAX_ATTEMPTS

sync_to_b2_once() {
  python3 "$REPO_ROOT/scripts/70_sync_760m_b2.py" \
    --paper-run-id "$PAPER_RUN_ID" \
    --exp-folder "$EXP_FOLDER" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --exp-dir "$REPO_ROOT/experiments" \
    --reports-root "$REPO_ROOT/reports/paper" \
    --b2-bucket "$B2_BUCKET" \
    --endpoint-url "$B2_ENDPOINT_URL" \
    --region "$AWS_DEFAULT_REGION" \
    --b2-prefix "$B2_ARTIFACT_PREFIX" \
    --run-log "$RUN_LOG" \
    --run-log "$SYNC_LOG"
}

start_sync_loop() {
  (
    while true; do
      {
        echo "[sync] $(date -u '+%Y-%m-%d %H:%M:%S UTC') starting B2 sync"
        sync_to_b2_once
        echo "[sync] $(date -u '+%Y-%m-%d %H:%M:%S UTC') finished B2 sync"
      } >>"$SYNC_LOG" 2>&1 || {
        rc=$?
        echo "[sync] $(date -u '+%Y-%m-%d %H:%M:%S UTC') B2 sync failed rc=$rc" >>"$SYNC_LOG"
      }
      sleep "$SYNC_INTERVAL_SECONDS"
    done
  ) &
  SYNC_PID="$!"
}

stop_sync_loop() {
  if [[ -n "$SYNC_PID" ]] && kill -0 "$SYNC_PID" 2>/dev/null; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
  fi
}

restore_latest_checkpoint_roots() {
  local remote_root="s3://$B2_BUCKET/$B2_ARTIFACT_PREFIX/checkpoints/$PAPER_RUN_ID"
  local local_root="$CHECKPOINT_ROOT/$EXP_FOLDER"
  mkdir -p "$local_root"

  local stage_dirs
  stage_dirs="$(aws s3 ls \
    "$remote_root/" \
    --endpoint-url "$B2_ENDPOINT_URL" \
    --region "$AWS_DEFAULT_REGION" 2>/dev/null | awk '$1 == "PRE" {print $2}')"

  if [[ -z "$stage_dirs" ]]; then
    echo "[bootstrap] no existing checkpoint roots found under $remote_root"
    return 0
  fi

  while IFS= read -r stage_dir; do
    [[ -z "$stage_dir" ]] && continue
    stage_dir="${stage_dir%/}"
    local remote_stage="$remote_root/$stage_dir"
    local local_stage="$local_root/$stage_dir"
    mkdir -p "$local_stage"

    aws s3 cp \
      "$remote_stage/latest.json" \
      "$local_stage/latest.json" \
      --endpoint-url "$B2_ENDPOINT_URL" \
      --region "$AWS_DEFAULT_REGION" \
      --only-show-errors

    local latest_fields
    latest_fields="$(python3 - "$local_stage/latest.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
step = str(payload.get("step", "")).strip()
path = str(payload.get("path", "")).strip()
metadata = str(payload.get("metadata_path", "")).strip()
print(step)
print(path)
print(metadata)
PY
)"

    local latest_step latest_path metadata_path
    latest_step="$(printf '%s\n' "$latest_fields" | sed -n '1p')"
    latest_path="$(printf '%s\n' "$latest_fields" | sed -n '2p')"
    metadata_path="$(printf '%s\n' "$latest_fields" | sed -n '3p')"

    echo "[bootstrap] restoring checkpoint root $stage_dir step=${latest_step:-unknown}"

    if [[ -n "$metadata_path" ]]; then
      mkdir -p "$local_stage/$(dirname "$metadata_path")"
      aws s3 cp \
        "$remote_stage/$metadata_path" \
        "$local_stage/$metadata_path" \
        --endpoint-url "$B2_ENDPOINT_URL" \
        --region "$AWS_DEFAULT_REGION" \
        --only-show-errors
    fi

    if [[ -n "$latest_path" ]]; then
      aws s3 sync \
        "$remote_stage/$latest_path" \
        "$local_stage/$latest_path" \
        --endpoint-url "$B2_ENDPOINT_URL" \
        --region "$AWS_DEFAULT_REGION" \
        --only-show-errors
    fi
  done <<< "$stage_dirs"
}

trap stop_sync_loop EXIT

echo "[bootstrap] syncing author checkpoints from B2"
aws s3 sync \
  "s3://$B2_BUCKET/$B2_ARTIFACT_PREFIX/author_checkpoints/760m_fa" \
  "$ARTIFACT_ROOT/760m_fa" \
  --endpoint-url "$B2_ENDPOINT_URL" \
  --region "$AWS_DEFAULT_REGION" \
  --only-show-errors

aws s3 sync \
  "s3://$B2_BUCKET/$B2_ARTIFACT_PREFIX/author_checkpoints/760m_e2e" \
  "$ARTIFACT_ROOT/760m_e2e" \
  --endpoint-url "$B2_ENDPOINT_URL" \
  --region "$AWS_DEFAULT_REGION" \
  --only-show-errors

echo "[bootstrap] restoring latest checkpoint tree from B2"
restore_latest_checkpoint_roots

echo "[bootstrap] fetching datasets from B2"
python3 "$REPO_ROOT/scripts/28_fetch_b2_dataset.py" \
  --dest-root "$DATA_ROOT" \
  --datasets dclm_filter_8k,books3 \
  --splits train,val \
  --b2-prefix "$B2_DATASET_PREFIX/$B2_DATASET_PACKAGE" \
  --force \
  --no-progress

cd "$REPO_ROOT"

if ! python3 -m pip --version >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y python3-pip
fi

echo "[bootstrap] syncing Python environment"
PIP_INSTALL_FLAGS=()
if python3 -m pip install --help 2>/dev/null | grep -q -- "--break-system-packages"; then
  PIP_INSTALL_FLAGS+=(--break-system-packages)
fi
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install "${PIP_INSTALL_FLAGS[@]}" -q uv
fi
if ! command -v aws >/dev/null 2>&1; then
  python3 -m pip install "${PIP_INSTALL_FLAGS[@]}" -q awscli
fi
uv sync

export PATH="$HOME/.local/bin:$PATH"
export TTT_ATTENTION_IMPLEMENTATION="$ATTN_IMPL"
if [[ -n "$SWA_PREFIX_ATTN_IMPL" ]]; then
  export TTT_SWA_PREFIX_ATTENTION_IMPLEMENTATION="$SWA_PREFIX_ATTN_IMPL"
fi

start_sync_loop

echo "[bootstrap] launching resumable ladder phase=$PHASE" | tee "$RUN_LOG"
ladder_rc=0
ladder_cmd=(
  uv run --exact python "$REPO_ROOT/scripts/66_run_760m_author_seed_ladder.py"
  --phase "$PHASE"
  --skip-existing
  --dclm-root "$DATA_ROOT/dclm_filter_8k"
  --books-root "$DATA_ROOT/books3"
  --attention-implementation "$ATTN_IMPL"
  --save-milestone-freq "$SAVE_MILESTONE_FREQ"
  --wandb-project none
  --wandb-entity none
  --allow-missing-fingerprints
  --b2-sync
  --b2-bucket "$B2_BUCKET"
  --b2-endpoint-url "$B2_ENDPOINT_URL"
  --b2-region "$AWS_DEFAULT_REGION"
  --b2-prefix "$B2_ARTIFACT_PREFIX"
  --run-log "$RUN_LOG"
  --run-log "$SYNC_LOG"
)
if [[ -n "$SWA_PREFIX_ATTN_IMPL" ]]; then
  ladder_cmd+=(--swa-prefix-attention-implementation "$SWA_PREFIX_ATTN_IMPL")
fi

if ! "${ladder_cmd[@]}" 2>&1 | tee -a "$RUN_LOG"; then
  ladder_rc=$?
fi

{
  echo "[bootstrap] running final B2 sync"
  sync_to_b2_once
} >>"$SYNC_LOG" 2>&1 || true

exit "$ladder_rc"
