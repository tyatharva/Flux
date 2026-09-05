# results/ml: the FNO emulator's scored artifacts

Produced by `ml/`; documented in `docs/emulator/training.md` and `docs/emulator/results.md`.
Checkpoints and predictions are on Hugging Face (`bin/fetch_assets.sh weights predictions`).

| path | what |
|---|---|
| `loader_audit.jsonl` | one line per corpus read: file, split, rows, `allow_test`; every test-split read carries `allow_test` |
| `phase1/summary.{tsv,md}`, `phase1/DECISIONS.md` | the exploration matrix and what it decided |
| `phase2/trials.tsv`, `phase2/study_summary.{json,md}` | the Optuna study |
| `haze/summary.md` | the post-Optuna round on the haze: cone gate and L1 variants |
| `final/seed*/run.json`, `final/final.json` | the five seeds of the final configuration |
| `eval/final_ensemble/`, `eval/final_seed*/`, `eval/floor/` | `ml/evaluate.py` output: `eval.{json,md}`, `per_record.tsv`; `floor/` is the two-window pair scored by the same evaluator |

`run.json` fields: `config`, `history` (per-epoch train/val loss), `best_epoch`, `val_loss` (own
transform space), `val_mse_ref` (the file's global asinh space, comparable across transforms),
`gap` (val/train), `composite`.
