#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/workspace/ttt-e2e-data}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/artifacts/author_checkpoints}"
PAPER_RUN_ID="${PAPER_RUN_ID:-protocol_r_760m_author_seed_v1}"
EXP_FOLDER="${EXP_FOLDER:-protocol_r_760m_author_seed_v1}"
SAVE_MILESTONE_FREQ="${SAVE_MILESTONE_FREQ:-2400}"
ATTN_IMPL="${ATTN_IMPL:-none}"
B2_ARTIFACT_PREFIX="${B2_ARTIFACT_PREFIX:-ttt-e2e-artifacts}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-600}"
GATE_LOG="${GATE_LOG:-/workspace/c1_gate.log}"
RUN_LOG="${RUN_LOG:-/workspace/c1_ladder.log}"
SYNC_LOG="${SYNC_LOG:-/workspace/c1_b2_sync.log}"
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

if [[ -d /root/.ssh ]]; then
  chmod 700 /root/.ssh || true
  if [[ -f /root/.ssh/authorized_keys ]]; then
    chmod 600 /root/.ssh/authorized_keys || true
  fi
fi

mkdir -p "$DATA_ROOT" "$CHECKPOINT_ROOT" "$ARTIFACT_ROOT"

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
    --run-log "$GATE_LOG" \
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

trap stop_sync_loop EXIT

echo "[bootstrap] syncing author checkpoints from B2"
aws s3 sync \
  "s3://$B2_BUCKET/ttt-e2e-artifacts/author_checkpoints/760m_fa" \
  "$ARTIFACT_ROOT/760m_fa" \
  --endpoint-url "$B2_ENDPOINT_URL" \
  --region "$AWS_DEFAULT_REGION" \
  --only-show-errors

aws s3 sync \
  "s3://$B2_BUCKET/ttt-e2e-artifacts/author_checkpoints/760m_e2e" \
  "$ARTIFACT_ROOT/760m_e2e" \
  --endpoint-url "$B2_ENDPOINT_URL" \
  --region "$AWS_DEFAULT_REGION" \
  --only-show-errors

echo "[bootstrap] restoring clean S2_ADAPT step-600 checkpoint from B2"
aws s3 sync \
  "s3://$B2_BUCKET/ttt-e2e-artifacts/checkpoints/$PAPER_RUN_ID/adapt-760m-e2e-8K-from-fa" \
  "$CHECKPOINT_ROOT/$EXP_FOLDER/adapt-760m-e2e-8K-from-fa" \
  --endpoint-url "$B2_ENDPOINT_URL" \
  --region "$AWS_DEFAULT_REGION" \
  --only-show-errors

echo "[bootstrap] fetching datasets from B2"
python3 "$REPO_ROOT/scripts/28_fetch_b2_dataset.py" \
  --dest-root "$DATA_ROOT" \
  --datasets dclm_filter_8k,books3 \
  --splits train,val \
  --b2-prefix "$B2_DATASET_PREFIX/paper_budget_760m_val-full" \
  --force \
  --no-progress

cd "$REPO_ROOT"

echo "[bootstrap] syncing Python environment"
uv sync

export PATH="$HOME/.local/bin:$PATH"
export TTT_ATTENTION_IMPLEMENTATION="$ATTN_IMPL"
start_sync_loop

echo "[bootstrap] running 5-step resume gate" | tee "$GATE_LOG"
gate_rc=0
if ! uv run --exact train \
  +deploy=interactive \
  +experiment=760m/pretrained/adapt-760m-e2e-8K-from-fa \
  training.exp_folder="$EXP_FOLDER" \
  training.exp_dir="$REPO_ROOT/experiments" \
  training.exp_name=adapt-760m-e2e-8K-from-fa \
  training.total_steps=605 \
  training.runtime_mode=jax_train \
  training.wandb_entity=none \
  training.wandb_project=none \
  training.wandb_key=env \
  deploy_paths.data.dclm_filter_8k="$DATA_ROOT/dclm_filter_8k" \
  deploy_paths.data.books3="$DATA_ROOT/books3" \
  deploy_paths.checkpoint="$CHECKPOINT_ROOT" \
  training.checkpoint_path="$CHECKPOINT_ROOT" \
  training.paper_run_id="$PAPER_RUN_ID" \
  training.stage_id=S2_ADAPT \
  training.run_id=adapt-760m-e2e-8K-from-fa \
  training.save_milestone_freq="$SAVE_MILESTONE_FREQ" \
  training.global_batch_size=8 \
  training.log_wandb=false \
  training.resume_checkpoint_path="$CHECKPOINT_ROOT/$EXP_FOLDER/adapt-760m-e2e-8K-from-fa" \
  training.resume_checkpoint_format=orbax \
  backend.distributed=false \
  training.load_part=all 2>&1 | tee -a "$GATE_LOG"; then
  gate_rc=$?
fi

{
  echo "[bootstrap] running post-gate B2 sync"
  sync_to_b2_once
} >>"$SYNC_LOG" 2>&1 || true

if [[ "$gate_rc" -ne 0 ]]; then
  exit "$gate_rc"
fi

echo "[bootstrap] gate passed, launching full C1 ladder" | tee "$RUN_LOG"
ladder_rc=0
if ! uv run --exact python "$REPO_ROOT/scripts/66_run_760m_author_seed_ladder.py" \
  --phase core \
  --dclm-root "$DATA_ROOT/dclm_filter_8k" \
  --books-root "$DATA_ROOT/books3" \
  --attention-implementation "$ATTN_IMPL" \
  --save-milestone-freq "$SAVE_MILESTONE_FREQ" \
  --wandb-project none \
  --wandb-entity none \
  --allow-missing-fingerprints \
  --b2-sync \
  --b2-bucket "$B2_BUCKET" \
  --b2-endpoint-url "$B2_ENDPOINT_URL" \
  --b2-region "$AWS_DEFAULT_REGION" \
  --b2-prefix "$B2_ARTIFACT_PREFIX" \
  --run-log "$GATE_LOG" \
  --run-log "$RUN_LOG" \
  --run-log "$SYNC_LOG" 2>&1 | tee -a "$RUN_LOG"; then
  ladder_rc=$?
fi

{
  echo "[bootstrap] running final B2 sync"
  sync_to_b2_once
} >>"$SYNC_LOG" 2>&1 || true

exit "$ladder_rc"
