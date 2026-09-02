# results/ml — the emulator's scored artifacts

Produced by the `ml/` package (see `ml/__init__.py`); write-up in
`docs/results/FNO_RESULT.md`. Tracked here: summaries (`.json`, `.tsv`, `.md`, `.png`).
Ignored (regenerable): `ckpt/`, `**/*.pt`, `*.db`, `cache/`.

| path | what |
|---|---|
| `loader_audit.jsonl` | one line per corpus read: file, split, rows, `allow_test`. **No line loads `test`.** |
| `cache/` | the val cone masks rebuilt from the file's rule (`ml/data.py:cone_masks`) |
| `runs/` | one-off training runs (`smoke*` are trainer smoke tests, not results) |
| `phase1/<run>/run.json` | Phase 1 exploration, one directory per configuration; `summary.{tsv,md}`, `gpu_util.csv`, `DECISIONS.md` |
| `phase2/` | the Optuna study: `optuna_<study>.db` (ignored), `trials.tsv`, `study_summary.{json,md}`, `trials/t*/run.json` |
| `final/` | the retrained final configuration (seeds, ensemble), `pred_val.npz` |
| `eval/<tag>/` | `ml/evaluate.py` output: `eval.{json,md}`, `per_record.tsv`, figures. `eval/floor/` is the two-window pair scored by the same evaluator |

`run.json` fields: `config`, `history` (per-epoch train/val loss), `best_epoch`, `val_loss`
(own transform space), `val_mse_ref` (the file's global asinh space, comparable across
transforms), `gap` (val/train), `composite` = geometric mean over the five production
metrics of median|err_FNO| / median|err_Kljun| on val (< 1 beats Kljun), `composite_north`
(N/NE/NW records only), `val_metrics`, `train_metrics`.
