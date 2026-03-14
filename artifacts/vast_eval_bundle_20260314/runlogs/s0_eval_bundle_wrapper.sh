#!/usr/bin/env bash
set -euo pipefail
cd /root/Reproducing-TTT-E2E
if [ ! -d .venv ]; then /root/.local/bin/uv venv --python 3.13 >/dev/null 2>&1; fi
. .venv/bin/activate
/root/.local/bin/uv sync --exact --python 3.13 >/dev/null
set -a
. /root/Reproducing-TTT-E2E/.env.runtime
set +a
/root/.local/bin/uv run --exact python scripts/54_run_protocol_r_parity_eval.py \
  --profile s0_pair \
  --paper-run-id protocol_r_125m_main_v1 \
  --repo-id "$HF_RESULTS_REPO" \
  --token "$HF_TOKEN" \
  --dclm-root /root/ttt-e2e-data/dclm_filter_8k \
  --books-root /root/ttt-e2e-data/books3
