# Reddit Deep Partial Monitoring

This project trains and evaluates two BERT-based offline contextual bandit systems for Reddit moderation:

- `message`: predicts from the current message only.
- `context`: predicts from the current message plus the previous messages in the same conversation.

At test time the policy can either keep a message, remove it, or ask a human moderator.

### Online direct-action policy

To let LinUCB choose actions directly and learn from simulated human feedback, run:

```bash
venv/bin/python scripts/train_eval.py \
  --data-path data/cad_v1_1.tsv \
  --bert-model /path/to/local/bert \
  --output-dir outputs/online_bandit \
  --algorithm linucb --policy online \
  --ridge 10 --human-cost 0.2 \
  --false-positive-cost 1 --false-negative-cost 25 \
  --exploration-rate 0.05 --alpha 0
```

This policy needs no calibration, quality targets or tuned thresholds. It first fits
the reward model using all labels in the training split, then processes the held-out
test messages in dataset order. For each message it chooses the action with the
highest estimated reward; with probability `--exploration-rate`, it instead asks a
human to obtain feedback. `--alpha` applies a conservative uncertainty penalty;
exploration is controlled separately. Human answers are simulated using test labels
and assumed correct. Only human-reviewed test messages update the model, after the
decision; labels for automatic decisions are used solely to calculate offline metrics.
Thus test results measure an adaptive sequential policy and depend on message order,
the random seed, and the assumption that human feedback is immediate and perfect.
The initial training phase is full-feedback, not a simulation of learning from scratch.
The event CSV records `exploration_human` and `feedback_observed`. The calibrated
and frozen reward policies remain available via `--policy calibrated` and `--policy reward`.

The experiment uses frozen BERT features and offline reward regression:

- BERT encodes the moderation state.
- The bandit can be `LinUCB` or `NeuralUCB`.
- Labels define all three action rewards, with configurable false-positive, false-negative and human-review costs.
- The default `calibrated` policy maps reward margins to removal probabilities, then selects separate keep/remove thresholds on held-out conversations.
- This is a full-feedback offline evaluation. Human decisions do not update the model during testing, and the cumulative test plots are not learning curves.

## Expected data

Put one CSV per conversation in a directory such as `data/conversations/`.
CSV files are discovered recursively by default, so nested folders under `--data-dir` are included.

Each CSV should have one row per message and at least:

- a text column, for example `text`, `body`, `message`, or `comment`
- a label column, for example `label`, `moderated`, `removed`, or `is_removed`

Labels are converted to binary moderation labels. Values like `1`, `true`, `removed`, `moderated`, `delete`, and `yes` are treated as removed/moderated.

### CAD v1.1 TSV

You can pass `data/cad_v1_1.tsv` directly to both scripts using `--data-path`
(an alias of `--data-dir`). A TSV file is automatically treated as CAD; use
`--data-format cad` to select this format explicitly. The CSV directory format
continues to work as before, and a single conversation CSV is also accepted.

CAD is converted to a binary abuse target for these moderation experiments:

| CAD `annotation_Primary` | Binary label |
|---|---:|
| `IdentityDirectedAbuse`, `AffiliationDirectedAbuse`, `PersonDirectedAbuse` | 1 |
| `Neutral`, `CounterSpeech`, `Slur` | 0 |

CAD's `Slur` category refers to non-hateful uses of slurs, as described in the
[CAD paper](https://aclanthology.org/2021.naacl-main.182/). This binary target
represents abuse annotations, rather than observed moderator removal decisions.
Multiple annotation rows for the same `(info_thread.id, info_id)` are merged into
one message; any positive category makes that message positive. Unknown labels
and inconsistent annotations raise an error.

The loader reads `meta_text`, replaces `[linebreak]` markers with newlines, and
excludes `exclude_*` rows, empty texts, and `[deleted]` / `[removed]` placeholders.
Messages are grouped by `info_thread.id` and sorted by `meta_created_utc`, with
titles before post bodies when timestamps match, then entry ID to break ties.
Context contains the preceding retained messages in that thread, limited by
`--context-window`; it is chronological history, not just the reply's ancestors.
Excluded messages do not enter the context.

By default, CAD's official `train/dev/test` assignments are preserved (`dev` is
validation), and validation is divided by conversation for calibration and
threshold selection. A thread spanning multiple official splits is rejected.
Use `--split-strategy random` to make a new conversation split with
`--train-fraction`, `--val-fraction`, and `--seed`. Those two fractions only apply
to random splitting. `--split-strategy provided` explicitly requires dataset
assignments; the default `auto` uses them when available and otherwise splits
CSV conversations randomly.

Audit the data without loading BERT:

```bash
venv/bin/python scripts/audit_data.py --data-path data/cad_v1_1.tsv
```

The local CAD v1.1 file produces 23,415 usable unique messages: 13,583 train,
4,525 validation, and 5,307 test. There are 4,119 binary positives. The audit also
reports 3,202 excluded annotation rows, 875 merged duplicate annotations, one
additional empty text, and one additional unavailable-text placeholder.

Run the same calibrated experiment on CAD, replacing the model path with your
local BERT directory:

```bash
venv/bin/python scripts/train_eval.py \
  --data-path data/cad_v1_1.tsv \
  --output-dir outputs/cad_calibrated_pm \
  --bert-model /path/to/local/bert \
  --algorithm linucb \
  --policy calibrated \
  --context-window 3 \
  --max-length 384 \
  --pooling mean \
  --ridge 10 \
  --human-cost 0.2 \
  --false-positive-cost 1 \
  --false-negative-cost 25 \
  --min-pipeline-recall 0.98 \
  --min-remove-precision 0.95 \
  --min-remove-examples 100 \
  --alpha 0
```

This runs both `message` and `context`. To test NeuralUCB, change
`--algorithm linucb` to `--algorithm neuralucb` and use a separate output directory,
such as `outputs/cad_calibrated_pm_neuralucb`. To run only the message model, add
`--systems message`.

`summary.json` records the resolved data format, split strategy, binary label
mapping, excluded rows, and merged annotations alongside the usual metrics.

## Install

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
```

## Local BERT Model

The code is designed for offline clusters such as Calcul Canada: it never downloads a model at runtime. Pass a local Hugging Face model directory with `--bert-model`.

Prepare the model on a machine with internet access, then copy the whole directory to the cluster:

```python
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"
output_dir = "models/bert-base-uncased"

AutoTokenizer.from_pretrained(model_name).save_pretrained(output_dir)
AutoModel.from_pretrained(model_name).save_pretrained(output_dir)
```

The local directory should contain files such as `config.json`, tokenizer files like `tokenizer.json` or `vocab.txt`, and weights like `model.safetensors` or `pytorch_model.bin`.
`--bert-model` must point to an existing local directory containing `config.json`;
Hub identifiers are not accepted. Relative paths are resolved from the job's
working directory. On a cluster, verify the directory is visible inside the
SLURM job. The loader reports missing paths and configurations before calling
Transformers, avoiding misleading Hugging Face repository-ID validation errors.

For a Toxic-BERT encoder comparison, use `model_name = "unitary/toxic-bert"`
and a separate local output directory in the preparation example above. Copy
the tokenizer and model together. `AutoModel` uses its encoder; the toxicity
classification head is not used, and the encoder remains frozen during the
bandit experiment.

If a local tokenizer has lost its padding-token declaration, the loader restores
the existing token identified by `pad_token_id` in the local `config.json`.
For BERT this is normally `[PAD]` (ID 0). This does not add vocabulary entries or
resize model embeddings. If the configured token is missing or invalid, loading
fails early with instructions to copy the matching tokenizer files.

Each example is encoded as one sequence, including any context, so the encoder
uses its default segment-zero IDs rather than forwarding tokenizer segment IDs.
Before sending a batch to the GPU, the code checks word IDs against the actual
embedding matrix and checks BERT's position limit. If a local tokenizer and
checkpoint do not match, this produces an explicit error instead of an invalid
embedding lookup on CUDA. Copy matching weights and tokenizer files together;
do not change vocabulary indices to suppress the error. For an unexplained CUDA
assert, restart the Python process and set `CUDA_LAUNCH_BLOCKING=1` before the
training command to locate the failing operation.

## Run an experiment

```bash
venv/bin/python scripts/train_eval.py \
  --data-dir data/conversations \
  --output-dir outputs/reddit_calibrated \
  --context-window 3 \
  --algorithm linucb \
  --policy calibrated \
  --bert-model models/bert-base-uncased \
  --max-length 384 \
  --pooling mean \
  --ridge 10 \
  --false-negative-cost 25 \
  --false-positive-cost 1 \
  --human-cost 0.2 \
  --class-weighting none \
  --alpha 0 \
  --min-pipeline-recall 0.98 \
  --min-remove-precision 0.95 \
  --min-remove-examples 100 \
  --selection-objective coverage
```

The train/validation/test split is by conversation. Only `train` fits the reward model.
Validation conversations are divided in two disjoint halves: one fits the unweighted sigmoid calibrator,
the other selects thresholds. Test labels are never used to calibrate or select thresholds.
Calibration needs both label classes; validation must contain at least two conversations.

The current message precedes context, with the most recent previous message first.
Right truncation preserves the beginning of the current message and spends the remaining token budget on context.
If the current message itself exceeds the budget, its tail is truncated.

Threshold selection maximizes automatic coverage subject to the chosen empirical pipeline recall,
minimum removal precision, and minimum removal support. Each automatic action group must cost no more
on tuning data than sending that group to humans. `--selection-objective cost` minimizes cost instead.
If no eligible group exists, that action remains disabled (`keep_max=-1` or `remove_min=2`).
Targets apply to tuning data and are not statistical guarantees on unseen data.
Pipeline recall assumes perfect human decisions.

To disable the minimum recall and removal-precision requirements, add
`--no-quality-constraints` to a calibrated run. This sets both targets to zero
and disables the recall-target sweep. The action-cost guard and
`--min-remove-examples` remain active, so automatic decisions can still be rare.
For a separate experiment allowing small removal groups, also set
`--min-remove-examples 1`. These changes do not impose a human-review budget.

You can reselect thresholds from an existing calibrated run without loading BERT
or retraining its bandit:

```bash
venv/bin/python scripts/retune_policy.py \
  --run-dir outputs/reddit_cad_hatebert_weight_1 \
  --output-dir outputs/reddit_cad_hatebert_no_quality \
  --no-quality-constraints
```

The output must be a new directory. The script selects thresholds using only
saved tuning-validation scores and replays saved test probabilities. It exports
events, cumulative metrics, validation tradeoffs and a summary recording the
source run and effective constraints. Repeated test comparisons are exploratory.

Useful options:

- `--text-column body`
- `--label-column label`
- `--no-recursive`: only read CSV files directly inside `--data-dir`
- `--train-fraction 0.7` and `--val-fraction 0.15`: conversation-level split sizes
- `--eval-split test`: log decisions only on the held-out test split
- `--eval-split all`: reward-policy diagnostic on all messages, including training; not a held-out evaluation
- `--systems message`: run only the message model to save compute
- `--calibration-fraction 0.5`: validation fraction used for calibration, with the remainder used for threshold tuning
- `--human-cost 0.2`: partial monitoring cost for asking a human
- `--false-negative-cost 25`: cost of keeping a message labeled for moderation
- `--false-positive-cost 1`: cost of removing a message labeled to be kept
- `--class-weighting balanced`: give rare positive examples more influence while fitting the bandit
- `--positive-weight 3`: manually set the positive-class weight; overrides `--class-weighting`
- `--algorithm linucb` or `--algorithm neuralucb`
- `--alpha 0`: no uncertainty penalty; `0.5` or `1` makes automatic decisions more conservative
- `--ridge 10`: LinUCB regularization
- `--pooling mean` or `--pooling pooler`: masked mean features or the previous pooler representation
- `--min-pipeline-recall 0.98`: tuning target for combined automatic and perfect-human recall
- `--recall-targets 0.99 0.98 0.97 0.95`: validation-only tradeoffs from the same fitted model
- `--neural-epochs 10`: neural reward-model epochs for NeuralUCB
- `--mc-samples 20`: MC-dropout samples used by NeuralUCB uncertainty
- `--bert-model path/to/model_dir`: local Hugging Face BERT-compatible model directory

## Outputs

The script writes:

- `metrics_summary.csv`: final metrics per system
- `events_message.csv` and `events_context.csv`: one test-time decision per message
- `metrics_message.csv` and `metrics_context.csv`: cumulative metrics over time
- `summary.json`: run configuration and metrics
- `validation_scores_message.csv` and `validation_scores_context.csv`: held-out raw scores, calibrated probabilities, uncertainty bounds, labels, and calibration/tuning membership
- `validation_tradeoffs_message.csv` and `validation_tradeoffs_context.csv`: thresholds and validation results for each recall target
- `plots/*.png`: accuracy, cumulative cost, human review rate, precision/recall/F1

In `summary.json`, `dataset.messages` is the number of loaded message rows, while `dataset.files_found` and `dataset.conversations` report the CSV/conversation counts. The `events_*.csv` files contain decisions for `dataset.eval_messages`, which is the test split by default.

Each system also records the selected thresholds, sigmoid parameters and tuning metrics under `policy`,
and the `all_keep`, `all_remove` and `all_human` baselines at the experiment's exact costs.
`probability_diagnostics` records test average precision, label prevalence and Brier score.
Test events include `probability_removed`, `probability_lower`, `probability_upper` and `decision_rule`.
Calibrated-policy `score_*` fields are expected negative costs; tuned thresholds, rather than a score argmax, determine the action.

`global_auto_remove_recall` divides correct automatic removals by **all** positive messages.
`pipeline_recall_perfect_human` also counts positive messages sent to humans.
`keep_error_rate` measures the positive fraction among automatically kept messages.
`false_positives` and `false_negatives` count automatic mistakes, while `cost_saving_vs_all_human`
compares test cost with human review of every message.
Legacy `recall_remove` and `f1_remove` remain restricted to automatically handled messages;
the explicit aliases `selective_recall_remove` and `selective_f1_remove` have the same meaning.

## Uncertainty and Offline Semantics

Full-feedback LinUCB has identical action design matrices, so a shared optimistic UCB bonus
cancels from an action argmax. The offline policies now use uncertainty conservatively instead:
LinUCB estimates reward-margin standard deviations using regression residuals and leverage,
and NeuralUCB uses MC-dropout margin deviations. The calibrated policy propagates
`score +/- alpha * std` through the sigmoid and keeps/removes only when the corresponding
probability bound crosses its threshold. These are heuristic bounds, not guaranteed confidence intervals.
The known human cost is fixed and has zero uncertainty.
`--policy reward` compares conservative reward estimates directly, without calibration.
The calibrated and reward policies do not implement online exploration; NeuralUCB does not use canonical gradient confidence bounds.

## Suggested Reddit Experiments

Start with the command above: no class weighting, ridge `10`, alpha `0`, masked mean pooling,
384 tokens, and recall `0.98`. Preserve the `25 / 1 / 0.2` cost assumptions across comparisons;
costs should reflect the application, not be adjusted to make a model look better.

The validation tradeoff CSV already compares recall targets `0.99`, `0.98`, `0.97`, and `0.95`
without another BERT pass. Inspect coverage, keep errors and cost together before selecting a target.
Only the target explicitly selected with `--min-pipeline-recall` is evaluated on test.

For separate representation/fit comparisons, try `--positive-weight 3`, then `--pooling pooler`,
changing one option at a time and using a fresh output directory for each run.
Use `--alpha 0.5` as an additional conservative comparison.
No reduction in human review is guaranteed if the frozen features do not separate the labels well.
Repeatedly choosing settings from the same test results would turn that test into another validation set.
