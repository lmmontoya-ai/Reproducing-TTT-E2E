# Reference Protocol R Batch Search

- Selected batch size: `8`
- Selection basis: `reference_train_fit_after_two_train_steps_and_checkpoint_before_full_holdout_eval`

| Batch | Status | RC | loss_ce | step |
| --- | --- | ---: | ---: | ---: |
| 32 | oom | 1 | — | — |
| 24 | oom | 1 | — | — |
| 16 | oom | 1 | — | — |
| 12 | failed_rc | 1 | — | — |
| 8 | fit_train_only | 0 | — | — |
