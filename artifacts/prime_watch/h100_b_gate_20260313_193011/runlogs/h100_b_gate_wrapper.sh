set -euo pipefail
cd /root/Reproducing-TTT-E2E
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.13 >/dev/null 2>&1
. .venv/bin/activate
uv sync --exact --python 3.13 >/dev/null 2>&1
set -a
source .env.runtime
set +a
uv run --exact python scripts/47_run_125m_split_batch.py --batch h100_b --paper-run-id protocol_r_125m_main_v1 --repo-id "$HF_RESULTS_REPO" --token "$HF_TOKEN" --dclm-root /root/ttt-e2e-data/dclm_filter_8k --books-root /root/ttt-e2e-data/books3 --wandb-entity "$WANDB_ENTITY" --wandb-project "$WANDB_PROJECT" --wandb-key env --skip-existing --stop-after-gates
