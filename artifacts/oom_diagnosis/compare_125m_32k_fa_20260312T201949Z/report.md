# 125M 32K FA Local-vs-Reference Comparison

Artifact dir: `/Users/lumontoya/omscs/cs8903/Reproducing-TTT-E2E/artifacts/oom_diagnosis/compare_125m_32k_fa_20260312T201949Z`

## Highest-priority findings
- `high` `train_runtime`: Local train path still replicates model state explicitly
  - The local runtime keeps the mutable Equinox state on a replicated sharding path, while the reference path relies on the mesh-oriented model/data flow. That can retain extra device copies or force less favorable compile decisions in long-context extension runs.
- `high` `train_runtime`: Local batch path still uses reshape-based helper instead of reference batch loading
  - The reference train loop builds global arrays from process-local data and only then rearranges the batch for data parallelism. The local helper hides that flow behind `to_data_parallel_batch`, which is a prime suspect for extra transient materialization.
- `medium` `train_runtime`: Local train loop still materializes richer per-step metric bookkeeping
  - This is unlikely to explain a 103 GB compile allocation by itself, but it does mean the local training function still carries more metric logic than the reference smoke path.
- `high` `loop_runtime`: Local train-step signature still differs from the reference axis contract
  - The local step vmaps over `(state, model, opt_state, batch)` in a different shape than the reference train step. That difference changes what the compiler sees as batched data versus replicated state and is a strong candidate for the extension OOM.
- `medium` `loop_runtime`: Local loop still emphasizes token-level metric aggregation inside the train/eval helpers
  - The reference evaluator collects metrics at loader boundaries. The local path still keeps token-NLL handling close to the loop helper, which is another place to check for retained arrays.
- `medium` `attention_runtime`: Local SWA/FA attention still has backend-conditional flash branches
  - The reference extension path assumes GPU execution and applies flash-attention constraints more directly. The local backend-conditional branches make the graph slightly less faithful and are worth ruling out in the compile-memory comparison.
- `high` `jax_utils`: Remat helper surface is still smaller than the reference utility stack
  - The reference utility layer exposes scan/remat chunking patterns that can materially change compile-memory behavior in long-context runs. The local helper parity is improved, but still not a one-to-one match at the utility boundary.

## Diff inventory
- `train_runtime`: +340 / -271 -> `train_runtime.diff`
- `loop_runtime`: +205 / -142 -> `loop_runtime.diff`
- `attention_runtime`: +331 / -132 -> `attention_runtime.diff`
- `jax_utils`: +304 / -206 -> `jax_utils.diff`
- `sharding_runtime`: +17 / -98 -> `sharding_runtime.diff`
- `experiment_ext_125m_32k_fa`: +0 / -0 -> `experiment_ext_125m_32k_fa.diff`
- `deploy_interactive`: +0 / -0 -> `deploy_interactive.diff`
