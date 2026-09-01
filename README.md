# Reddit Deep Partial Monitoring

This project trains and evaluates two BERT-based offline contextual bandit systems for Reddit moderation:

- `message`: predicts from the current message only.
- `context`: predicts from the current message plus the previous messages in the same conversation.

At test time the policy can either keep a message, remove it, or ask a human moderator.

The learning problem is modeled as an offline contextual bandit:

- BERT encodes the moderation state.
- The bandit can be `LinUCB` or `NeuralUCB`.
- Labels define the partial-monitoring rewards: wrong automatic decisions cost `1`, human review costs `--human-cost`, and correct automatic decisions cost `0`.

## Expected data

Put one CSV per conversation in a directory such as `data/conversations/`.
CSV files are discovered recursively by default, so nested folders under `--data-dir` are included.

Each CSV should have one row per message and at least:

- a text column, for example `text`, `body`, `message`, or `comment`
- a label column, for example `label`, `moderated`, `removed`, or `is_removed`

Labels are converted to binary moderation labels. Values like `1`, `true`, `removed`, `moderated`, `delete`, and `yes` are treated as removed/moderated.

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

## Run an experiment

```bash
venv/bin/python scripts/train_eval.py \
  --data-dir data/conversations \
  --output-dir outputs/reddit_pm \
  --context-window 6 \
  --algorithm linucb \
  --bert-model models/bert-base-uncased \
  --false-negative-cost 10 \
  --class-weighting balanced
```

Useful options:

- `--text-column body`
- `--label-column label`
- `--no-recursive`: only read CSV files directly inside `--data-dir`
- `--train-fraction 0.7` and `--val-fraction 0.15`: conversation-level split sizes
- `--eval-split test`: log decisions only on the held-out test split
- `--eval-split all`: log decisions on every loaded message after fitting on train+val
- `--human-cost 0.2`: partial monitoring cost for asking a human
- `--false-negative-cost 10`: cost of keeping a toxic message
- `--false-positive-cost 1`: cost of removing a non-toxic message
- `--class-weighting balanced`: give rare positive examples more influence while fitting the bandit
- `--positive-weight 50`: manually set the toxic-class weight; overrides `--class-weighting`
- `--algorithm linucb` or `--algorithm neuralucb`
- `--alpha 1.0`: UCB exploration strength
- `--neural-epochs 10`: neural reward-model epochs for NeuralUCB
- `--mc-samples 20`: MC-dropout samples used by NeuralUCB uncertainty
- `--bert-model path/to/model_dir`: local Hugging Face BERT-compatible model directory

## Outputs

The script writes:

- `metrics_summary.csv`: final metrics per system
- `events_message.csv` and `events_context.csv`: one test-time decision per message
- `metrics_message.csv` and `metrics_context.csv`: cumulative metrics over time
- `summary.json`: run configuration and metrics
- `plots/*.png`: accuracy, cumulative cost, human review rate, precision/recall/F1

In `summary.json`, `dataset.messages` is the number of loaded message rows, while `dataset.files_found` and `dataset.conversations` report the CSV/conversation counts. The `events_*.csv` files contain decisions for `dataset.eval_messages`, which is the test split by default.

## Rare Toxicity

If toxicity is very rare, a symmetric cost matrix usually makes `keep` optimal too often. Increase `--false-negative-cost` to make missed toxicity expensive, lower `--human-cost` if human review is acceptable, and keep `--class-weighting balanced` enabled.

Example:

```bash
venv/bin/python scripts/train_eval.py \
  --data-dir data/conversations \
  --output-dir outputs/reddit_pm_recall \
  --algorithm linucb \
  --bert-model models/bert-base-uncased \
  --false-negative-cost 25 \
  --false-positive-cost 1 \
  --human-cost 0.2 \
  --alpha 2.0
```
