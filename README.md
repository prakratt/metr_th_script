# METR Time Horizon: Standalone Replication Script

A single-file Python script that computes [METR's task-completion time horizons](https://metr.org/time-horizons/) for AI models. Given a set of tasks (with human completion times) and model success rates, it produces the same results as METR's multi-stage DVC pipeline — verified with **exact agreement on all 40 point estimates** against their published benchmark.

> **What is a time horizon?** The task duration (in human-expert minutes) at which an AI model succeeds with a given probability. A 50% time horizon of 60 minutes means the model succeeds half the time on 1-hour tasks. See the [paper](https://arxiv.org/abs/2503.14499) for details.

## Quick Start

### 1. Install dependencies

```bash
pip install numpy pandas scikit-learn pyyaml joblib
# Optional (for plotting):
pip install matplotlib plotly
```

### 2. Prepare your data

Create a CSV or JSONL file with one row per *evaluation run*:

```csv
task_id,task_family,agent,human_minutes,score_binarized
login_bypass,security,GPT-4o,45.0,1
login_bypass,security,GPT-4o,45.0,0
train_cifar,ml_tasks,GPT-4o,180.0,0
fix_bug_react,debugging,GPT-4o,25.0,1
login_bypass,security,Claude-3.5,45.0,1
login_bypass,security,Claude-3.5,45.0,1
train_cifar,ml_tasks,Claude-3.5,180.0,1
fix_bug_react,debugging,Claude-3.5,25.0,1
```

| Column | Description | Required |
|--------|-------------|----------|
| `task_id` | Unique identifier for the task | Yes |
| `task_family` | Group of related tasks (e.g., "security", "debugging"). Used for diversity weighting and hierarchical bootstrap. If you don't have families, use `--auto-family` to generate them from `task_id` prefixes. | Yes (or use `--auto-family`) |
| `agent` | Model/agent name | Yes |
| `human_minutes` | How long the task takes a human expert, in minutes | Yes |
| `score_binarized` | Did the model succeed? `1` = yes, `0` = no. Continuous scores in [0, 1] are also supported. | Yes |
| `run_id` | Unique run identifier | No (auto-generated if absent) |

**Multiple runs per (task, agent) pair are expected** — the script needs repeated evaluations to estimate success rates.

### 3. Run

```bash
python compute_time_horizon.py compute \
  --input my_eval_data.csv \
  --output results.yaml
```

This will:
- Compute inverse-sqrt task-family weights
- Run 1000 bootstrap iterations for confidence intervals
- Fit logistic regression per agent
- Output p50 and p80 time horizons with 95% CIs

To save memory (recommended for laptops), reduce bootstrap iterations and use single-threaded mode:

```bash
python compute_time_horizon.py compute \
  --input my_eval_data.csv \
  --output results.yaml \
  --n-bootstrap 100 \
  --n-jobs 1
```

### 4. View results

Open `results.yaml` directly, or generate the interactive HTML comparison page:

```bash
# If you also have METR's benchmark results for comparison:
python plot_comparison.py

# Then open in your browser:
open time_horizon_comparison.html
```

The HTML page includes interactive log/linear scale plots, a validation table, CI comparison, and full methodology documentation — all in one file.

## Adapting Your Data Format

The script auto-detects many common column names:

| Your column name | Auto-detected as |
|------------------|-----------------|
| `model`, `alias`, `model_name`, `system` | `agent` |
| `task`, `problem_id`, `question_id` | `task_id` |
| `family`, `category`, `domain`, `suite` | `task_family` |
| `duration_minutes`, `human_time`, `expert_minutes` | `human_minutes` |
| `human_hours`, `duration_hours` | `human_minutes` (auto-converted) |
| `score`, `success`, `pass`, `correct`, `solved` | `score_binarized` |

If your columns don't match, use `--column-map`:

```bash
python compute_time_horizon.py compute \
  --input data.csv \
  --column-map "my_model_col=agent,hours_taken=human_hours" \
  --auto-family \
  --output results.yaml
```

Or use the `convert` subcommand to preview how your data will be mapped:

```bash
python compute_time_horizon.py convert \
  --input raw_data.csv \
  --column-map "model=agent,duration_hours=human_hours" \
  --auto-family \
  --preview
```

## How the Time Horizon is Measured

The script fits a **logistic regression** on `log2(human_minutes)` vs. binary success, then inverts the fitted curve to find when P(success) = 50%:

```
P(success | x) = sigmoid(beta_0 + beta_1 * log2(x))

50% time horizon: T_50 = 2^(-beta_0 / beta_1)
```

Key details:
- **Task weighting**: Inverse-sqrt family weighting prevents large task families from dominating
- **Bootstrap CIs**: Hierarchical resampling (family -> task -> run) for honest uncertainty estimates
- **Doubling time**: Exponential trend through SOTA agents measures capability growth rate

See [METHODOLOGY.md](METHODOLOGY.md) for the full mathematical treatment, or open the Methodology tab in the HTML results page.

## All Commands

```bash
# Compute time horizons
python compute_time_horizon.py compute --input data.csv --output results.yaml

# With release dates (enables trend/doubling time analysis)
python compute_time_horizon.py compute \
  --input data.csv \
  --release-dates dates.yaml \
  --output results.yaml

# Convert non-standard data formats
python compute_time_horizon.py convert --input raw.csv --output clean.csv --auto-family --preview

# Validate against known results
python compute_time_horizon.py validate \
  --input runs.jsonl \
  --expected benchmark_results.yaml \
  --release-dates dates.yaml

# Run built-in self-tests
python compute_time_horizon.py selftest

# Generate a static plot (requires matplotlib)
python compute_time_horizon.py plot \
  --input results.yaml \
  --release-dates dates.yaml \
  --output plot.png
```

### Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--n-bootstrap` | 1000 | Bootstrap iterations. Use 100 for faster/lower-memory runs. |
| `--n-jobs` | -1 (all cores) | Parallel workers. Use 1 to minimize memory. |
| `--weighting` | `invsqrt_task_weight` | `equal_task_weight` or `invsqrt_task_weight` |
| `--success-percents` | `50,80` | Which horizons to compute |
| `--confidence-level` | 0.95 | CI confidence level |
| `--format` | `yaml` | Output format: `yaml`, `json`, or `csv` |
| `--score-col` | `score_binarized` | Column name for scores |
| `--auto-family` | off | Auto-generate `task_family` from `task_id` prefixes |
| `--column-map` | none | Rename columns, e.g. `model=agent,hours=human_hours` |

## Files

| File | Description |
|------|-------------|
| `compute_time_horizon.py` | Standalone computation script (~1500 lines) |
| `test_time_horizon.py` | Test suite (49 tests) |
| `METHODOLOGY.md` | Full mathematical methodology and API reference |
| `plot_comparison.py` | Interactive HTML comparison page generator |
| `benchmark_results_1_1.yaml` | METR's published benchmark results (reference) |
| `full_results.yaml` | Our computed results (100 bootstrap iterations) |
| `time_horizon_comparison.html` | Interactive results page — open in browser |

## Validation

The script reproduces METR's published results exactly:

- **40/40 point estimates match** (p50, p80, average score for all 20 v1.1 agents)
- **Doubling time: 128.7 days** (exact match to METR's published value)
- CIs are consistent between our 100-sample and METR's 1000-sample bootstrap

Run `python compute_time_horizon.py selftest` to verify the core math with built-in synthetic tests.

## References

- METR Time Horizons website: https://metr.org/time-horizons/
- Paper: Kwa et al., "Measuring AI Ability to Complete Long Software Tasks" ([arXiv:2503.14499](https://arxiv.org/abs/2503.14499))
- Original analysis code: https://github.com/METR/eval-analysis-public