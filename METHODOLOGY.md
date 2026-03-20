# METR Time Horizon: Mathematical Methodology & API Reference

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [Step-by-Step Pipeline](#3-step-by-step-pipeline)
4. [API Reference](#4-api-reference)
5. [Input/Output Formats](#5-inputoutput-formats)
6. [Worked Example](#6-worked-example)
7. [Simplifications & Limitations](#7-simplifications--limitations)

---

## 1. Overview

A **time horizon** answers the question: *"For tasks that take human experts T minutes, what fraction of the time does model M succeed?"*

More precisely, the **p%-time horizon** of model M is the task duration T (in human-expert minutes) at which M is predicted to succeed with probability p%. For example:

- A **50% time horizon of 60 minutes** means the model succeeds ~50% of the time on tasks that take humans 1 hour.
- An **80% time horizon of 10 minutes** means the model succeeds ~80% on 10-minute tasks.

Longer time horizons indicate more capable models. The trend in time horizons over model release dates measures the rate of AI capability improvement.

**Reference**: Kwa et al., "Measuring the Capability Frontier" ([arXiv:2503.14499](https://arxiv.org/abs/2503.14499))

---

## 2. Mathematical Foundation

### 2.1 The Logistic Model

We model the relationship between task duration and success probability using **logistic regression** in log-time space:

$$P(\text{success} \mid x) = \sigma(\beta_0 + \beta_1 \cdot \log_2(x))$$

where:
- $x$ = `human_minutes` (task duration for human experts)
- $\log_2(x)$ = log-base-2 of the task duration (the feature)
- $\beta_0$ = intercept
- $\beta_1$ = coefficient (typically negative: longer tasks are harder)
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ = the logistic sigmoid function

**Why log2?** Task difficulty grows roughly exponentially with duration. A 1-unit increase in $\log_2(x)$ corresponds to doubling the task length, making the coefficient interpretable: $\beta_1$ captures how much log-odds of success change when the task doubles in length.

### 2.2 Time Horizon Extraction

Given the fitted model $(\beta_0, \beta_1)$, the **p%-time horizon** is found by solving:

$$p/100 = \sigma(\beta_0 + \beta_1 \cdot \log_2(T_p))$$

Inverting the sigmoid:

$$\log\left(\frac{p/100}{1 - p/100}\right) = \beta_0 + \beta_1 \cdot \log_2(T_p)$$

$$\log_2(T_p) = \frac{\log(p/(100-p)) - \beta_0}{\beta_1}$$

$$\boxed{T_p = 2^{\frac{\ln(p/(100-p)) - \beta_0}{\beta_1}}}$$

For the common cases:
- **p50**: $\ln(1) = 0$, so $T_{50} = 2^{-\beta_0/\beta_1}$
- **p80**: $\ln(4) \approx 1.386$, so $T_{80} = 2^{(1.386 - \beta_0)/\beta_1}$

Since $\beta_1 < 0$ (longer tasks are harder), we always have $T_{80} < T_{50}$ — the model can only maintain 80% success on shorter tasks than 50%.

### 2.3 Task Weighting

Tasks come in **families** (groups of related tasks). Without weighting, families with many tasks would dominate the regression. Two weighting schemes are available:

#### Equal Task Weight

Each task contributes equally, regardless of family size or number of evaluation runs:

$$w_i^{\text{equal}} = \frac{1}{n_{\text{runs}}(t_i)}$$

where $n_{\text{runs}}(t_i)$ is the number of runs for task $t_i$. Weights are then normalized so $\sum_i w_i = 1$ per agent.

#### Inverse-Square-Root Task Weight (default)

Additionally downweights tasks from large families to promote diversity:

$$w_i^{\text{invsqrt}} = \frac{1}{n_{\text{runs}}(t_i)} \cdot \frac{1}{\sqrt{n_{\text{tasks}}(f_i)}}$$

where $n_{\text{tasks}}(f_i)$ is the number of tasks in family $f_i$. Again normalized to $\sum_i w_i = 1$.

**Intuition**: If family A has 20 tasks and family B has 5, each task in A gets weight $\propto 1/\sqrt{20} \approx 0.224$ while each task in B gets $\propto 1/\sqrt{5} \approx 0.447$. This prevents a single large family from dominating the fit.

### 2.4 Handling Continuous Scores

Some tasks have continuous scores $y_i \in (0, 1)$ rather than binary $\{0, 1\}$. These are handled by **splitting** each fractional observation into two weighted binary observations:

For observation $(x_i, y_i, w_i)$ where $0 < y_i < 1$:
- Create $(x_i, 0, w_i \cdot (1 - y_i))$ — "failure" component
- Create $(x_i, 1, w_i \cdot y_i)$ — "success" component

This preserves the weighted average: $\text{E}[y] = w_i \cdot y_i \cdot 1 + w_i \cdot (1-y_i) \cdot 0 = w_i \cdot y_i$.

### 2.5 Regularization

The logistic regression uses L2 (ridge) regularization via sklearn's `C` parameter:

$$C = \frac{1}{\lambda} = \frac{1}{10^{-5}} = 100{,}000$$

This is very weak regularization — nearly unregularized — but prevents numerical instability when data is near-separable.

### 2.6 Bootstrap Confidence Intervals

Confidence intervals are estimated via **hierarchical bootstrap** with 1,000 iterations:

**Algorithm** (for each iteration $b = 1, \ldots, B$):

1. **Level 1 (Families)**: Resample task families with replacement. If there are $F$ unique families, draw $F$ families with replacement.
2. **Level 2 (Tasks)**: Within each resampled family, resample tasks with replacement.
3. **Level 3 (Runs)**: Within each (task, agent) pair, resample individual runs with replacement.
4. **Fit**: Fit logistic regression on the resampled data (weights are NOT renormalized).
5. **Extract**: Compute $T_{50}^{(b)}$ and $T_{80}^{(b)}$ from the bootstrapped model.

**CI extraction**: The 95% confidence interval is:

$$\text{CI}_{95\%} = \left[Q_{0.025}\left(\{T_p^{(b)}\}_{b=1}^B\right),\; Q_{0.975}\left(\{T_p^{(b)}\}_{b=1}^B\right)\right]$$

where $Q_\alpha$ denotes the $\alpha$-quantile.

**Why hierarchical?** If we only resampled runs, we'd underestimate uncertainty from task selection. By resampling at the family and task levels, we capture the variance in "which tasks happened to be in our benchmark."

**Seeding**: Iteration $b$ uses seed $42 + b$ via `np.random.default_rng(42 + b)` for exact reproducibility.

### 2.7 Doubling Time (Trend Analysis)

Given SOTA (state-of-the-art) models' p50 time horizons and their release dates, we fit an exponential trend:

$$\ln(T_{50}) = \alpha + \gamma \cdot d$$

where $d$ = date (as a numeric value, days since epoch). The **doubling time** is:

$$\tau_{\text{double}} = \frac{\ln 2}{\gamma}$$

This measures how many days it takes for the SOTA time horizon to double.

**SOTA determination**: A model is SOTA if, at its release date, its p50 time horizon $\geq$ the highest p50 of any model released on or before that date.

**Bootstrap CIs on doubling time**: For each bootstrap sample, use the SOTA agents' bootstrapped p50 values to fit the trend and extract the doubling time. The CI is the [2.5%, 97.5%] quantile of the resulting distribution (excluding non-positive doubling times).

---

## 3. Step-by-Step Pipeline

```
Input: runs.{csv,jsonl}
  |
  v
[1] Load & Validate Data
  - Accept various column names via aliases or --column-map
  - Normalize to: task_id, task_family, agent, human_minutes, score, run_id
  |
  v
[2] Compute Task Weights (per agent)
  - equal_task_weight OR invsqrt_task_weight
  - Weights sum to 1.0 per agent
  |
  v
[3] Bootstrap (1000 iterations, parallel)
  - Hierarchical resample: family -> task -> run
  - For each sample: fit logistic per agent, extract horizons
  - Output: DataFrame of bootstrapped horizons
  |
  v
[4] Point-Estimate Regression (per agent)
  - X = log2(human_minutes), y = scores, w = weights
  - Handle fractional y by splitting into weighted binary pairs
  - Fit LogisticRegression(C=100000)
  - Extract p50, p80 horizons
  - Extract CIs from bootstrap quantiles
  |
  v
[5] Trend Analysis (optional, requires release dates)
  - Identify SOTA agents (monotonic p50 frontier)
  - Fit: log(p50) ~ date (linear regression)
  - Doubling time = ln(2) / slope
  - Bootstrap CIs on doubling time
  |
  v
[6] Output: YAML / JSON / CSV
  - Per-agent: average_score, p50, p80, CIs, is_sota
  - Global: doubling_time with CIs
```

---

## 4. API Reference

### Data Loading

#### `load_run_data(filepath, score_col, column_map, auto_family) -> DataFrame`

Loads evaluation data from CSV, JSONL, JSON, or TSV files with flexible column name resolution.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str` | required | Path to data file |
| `score_col` | `str` | `"score_binarized"` | Score column name. Aliases: `"score"`, `"success"`, `"pass"`, `"correct"`, `"solved"` |
| `column_map` | `dict` | `None` | Explicit column renaming, e.g. `{"my_model": "agent"}` |
| `auto_family` | `bool` | `False` | Auto-generate `task_family` from `task_id` prefixes if missing |

**Auto-detected column aliases:**
| Target Column | Accepted Aliases |
|---------------|-----------------|
| `agent` | `alias`, `model_name`, `model`, `agent_name`, `system` |
| `task_id` | `task`, `problem_id`, `question_id`, `item_id` |
| `task_family` | `family`, `category`, `domain`, `task_group`, `suite` |
| `human_minutes` | `human_time`, `duration_minutes`, `time_minutes`, `expert_minutes` |
| `score_binarized` | `score`, `success`, `pass`, `correct`, `solved`, `result` |

**Auto-detected time conversions:**
- `human_hours` / `duration_hours` -> multiplied by 60 to get minutes
- `human_seconds` / `duration_seconds` -> divided by 60 to get minutes

**Returns:** DataFrame with standardized columns: `task_id`, `task_family`, `agent`, `human_minutes`, `<score_col>`, `run_id`.

---

#### `load_release_dates(filepath) -> dict[str, str]`

Loads model release dates from a YAML file.

**Accepted formats:**
```yaml
# Format 1: nested under "date" key
date:
  "Model A": "2024-01-15"
  "Model B": "2024-06-01"

# Format 2: flat mapping
"Model A": "2024-01-15"
"Model B": "2024-06-01"
```

**Returns:** Dict mapping agent name to `"YYYY-MM-DD"` string.

---

### Weight Computation

#### `compute_task_weights(df_agent, scheme) -> Series`

Computes per-run sample weights for a single agent.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df_agent` | `DataFrame` | required | Runs for one agent. Must have `task_id`, `task_family`, `run_id`. |
| `scheme` | `str` | `"invsqrt_task_weight"` | `"equal_task_weight"` or `"invsqrt_task_weight"` |

**Mathematical formulas:**

*equal_task_weight*: $w_i = \frac{1}{|\text{runs in task}_i|}$, then $w_i \leftarrow w_i / \sum_j w_j$

*invsqrt_task_weight*: $w_i = \frac{1}{|\text{runs in task}_i|} \cdot \frac{1}{\sqrt{|\text{tasks in family}_i|}}$, then $w_i \leftarrow w_i / \sum_j w_j$

**Returns:** Series of weights, summing to 1.0, aligned with `df_agent` index.

---

#### `add_weight_column(df, scheme) -> DataFrame`

Applies `compute_task_weights` per agent and adds a `"weight"` column.

---

### Core Logistic Regression

#### `fit_logistic(X, y, sample_weight, regularization, ensure_weights_sum_to_1) -> LogisticRegression`

Fits a weighted logistic regression with support for continuous (fractional) scores.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `ndarray (n,1)` | required | Features. Typically `log2(human_minutes)`. |
| `y` | `ndarray (n,)` | required | Scores in [0, 1]. Binary or continuous. |
| `sample_weight` | `ndarray (n,)` | required | Per-sample weights. |
| `regularization` | `float` | `1e-5` | L2 regularization ($C = 1/\lambda$). |
| `ensure_weights_sum_to_1` | `bool` | `True` | Assert weight normalization. Set `False` for bootstrap. |

**Fractional y handling:** For each observation with $0 < y_i < 1$:
- Split into $(X_i, 0, w_i \cdot (1-y_i))$ and $(X_i, 1, w_i \cdot y_i)$
- Preserves $\mathbb{E}[y] = \sum_i w_i y_i$

**Returns:** Fitted `sklearn.linear_model.LogisticRegression` with attributes:
- `model.coef_[0][0]` = $\beta_1$ (coefficient on $\log_2(\text{minutes})$)
- `model.intercept_[0]` = $\beta_0$ (intercept)

---

#### `get_horizon_at_percent(model, success_percent) -> float`

Extracts the time horizon from a fitted logistic model.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `LogisticRegression` | Fitted logistic model |
| `success_percent` | `int` | Target success rate (e.g., 50) |

**Formula:**
$$T_p = 2^{\frac{\ln(p/(100-p)) - \beta_0}{\beta_1}}$$

**Returns:** Time horizon in minutes.

---

### Per-Agent Computation

#### `compute_agent_horizon(human_minutes, scores, weights, agent_name, ...) -> dict`

Computes all time horizon metrics for a single agent.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `human_minutes` | `ndarray` | required | Task durations (minutes) |
| `scores` | `ndarray` | required | Success scores in [0, 1] |
| `weights` | `ndarray` | required | Sample weights |
| `agent_name` | `str` | required | Agent/model name |
| `regularization` | `float` | `1e-5` | Logistic regularization |
| `success_percents` | `list[int]` | `[50, 80]` | Percentiles to compute |
| `confidence_level` | `float` | `0.95` | CI confidence level |
| `bootstrap_horizons` | `DataFrame` | `None` | Bootstrap results for CIs |

**Processing steps:**
1. Transform: $X = \log_2(\text{human\_minutes})$
2. Compute weighted average score
3. Handle all-zeros edge case (return 0 horizons)
4. Fit logistic regression
5. Extract horizons at each success percent
6. If bootstrap results provided, extract CI bounds via quantiles

**Returns:** Dict with keys:
```python
{
    "agent": str,
    "average_score": float,
    "coefficient": float,  # beta_1
    "intercept": float,    # beta_0
    "p50": float,          # minutes
    "p50_ci_low": float,
    "p50_ci_high": float,
    "p80": float,
    "p80_ci_low": float,
    "p80_ci_high": float,
}
```

---

### Bootstrap

#### `bootstrap_sample(data, categories, rng) -> DataFrame`

Performs hierarchical bootstrap resampling.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `DataFrame` | Evaluation runs |
| `categories` | `list[str]` | Hierarchy: `["task_family", "task_id", "run_id"]` |
| `rng` | `np.random.Generator` | Random number generator |

**Algorithm:**
```
For each level in [task_family, task_id]:
  For each parent group:
    Get unique values at this level
    Resample N values from N with replacement
    Keep all children of resampled values

For run_id level:
  For each (task, agent) pair:
    Resample runs with replacement within the pair
```

**Returns:** Resampled DataFrame (may have different length than input due to resampling).

---

#### `compute_bootstrap_horizons(data, n_bootstrap, ...) -> DataFrame`

Runs the full bootstrap computation in parallel.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` | required | Full dataset with weights |
| `n_bootstrap` | `int` | `1000` | Number of iterations |
| `regularization` | `float` | `1e-5` | Logistic regularization |
| `weights_col` | `str` | `"weight"` | Weight column name |
| `success_percents` | `list[int]` | `[50, 80]` | Percentiles to bootstrap |
| `score_col` | `str` | `"score_binarized"` | Score column |
| `seed` | `int` | `42` | Base seed (iter i uses seed+i) |
| `n_jobs` | `int` | `-1` | Parallel workers (-1=auto) |

**Key details:**
- Uses `joblib.Parallel` with batch size 10
- Weights are NOT renormalized after bootstrap resampling
- Agents with no score variation in a sample are skipped
- Seed for iteration $i$: `np.random.default_rng(42 + i)`

**Returns:** DataFrame with shape `(n_bootstrap, n_agents * n_percents)`. Column names: `"{agent}_p{percent}"`.

---

### Trend Analysis

#### `determine_sota_agents(agent_results, release_dates, ...) -> list[str]`

Identifies state-of-the-art agents based on p50 at release time.

**Algorithm:**
1. Sort all agents by release date
2. Scan chronologically, tracking the maximum p50 seen
3. An agent is SOTA if its p50 >= the running maximum

Agents released on the same date are evaluated together. An agent that ties the current maximum is considered SOTA.

**Returns:** List of SOTA agent names in chronological order.

---

#### `compute_doubling_time(p50s, dates) -> (float, float)`

Fits an exponential trend to compute how fast time horizons are doubling.

**Model:** $\ln(T_{50}) = \alpha + \gamma \cdot d$ where $d$ is date in numeric form.

**Doubling time:** $\tau = \frac{\ln 2}{\gamma}$ (in days)

**Returns:** `(doubling_time_days, r_squared)`

---

#### `compute_trend_with_ci(agent_results, bootstrap_horizons, release_dates, ...) -> dict`

Computes doubling time with bootstrap confidence intervals.

**Process:**
1. Determine SOTA agents from point estimates
2. For each bootstrap sample: fit trendline through SOTA agents' bootstrapped p50s
3. Extract doubling time; keep only positive values
4. CI = quantiles of the doubling time distribution

**Returns:**
```python
{"point_estimate": float, "ci_low": float, "ci_high": float}
```

---

### Output Formatting

#### `format_results(agent_results, release_dates, doubling_time_stats, benchmark_name, fmt) -> str`

Formats results in YAML, JSON, or CSV.

**YAML output structure** (matches METR's `benchmark_results.yaml`):
```yaml
benchmark_name: METR-Horizon-v1.1
doubling_time_in_days:
  from_2023_on:
    point_estimate: 128.744
    ci_low: 105.26
    ci_high: 157.336
results:
  model_key:
    benchmark_name: METR-Horizon-v1.1
    metrics:
      average_score:
        estimate: 0.558217
      is_sota: true
      p50_horizon_length:
        estimate: 60.388937
        ci_low: 33.385879
        ci_high: 107.302719
      p80_horizon_length:
        estimate: 12.09179
        ci_low: 4.538607
        ci_high: 29.331824
    release_date: '2025-02-24'
```

---

### Plotting

#### `plot_horizons(agent_results, release_dates, success_percent, output_path, show_trendline)`

Generates a scatter plot of time horizons vs. release dates.

- Log y-axis (minutes)
- Error bars from bootstrap CIs
- Exponential trendline through SOTA agents
- Annotation with doubling time

Requires `matplotlib`.

---

## 5. Input/Output Formats

### Standard Input Format

**CSV:**
```csv
task_id,task_family,agent,human_minutes,score_binarized,run_id
debug_server/v1,debug_server,GPT-4o,45.0,1,run_001
debug_server/v1,debug_server,GPT-4o,45.0,0,run_002
train_model/v3,ml_tasks,GPT-4o,180.0,0,run_003
```

**JSONL:**
```json
{"task_id":"debug_server/v1","task_family":"debug_server","agent":"GPT-4o","human_minutes":45.0,"score_binarized":1,"run_id":"run_001"}
{"task_id":"debug_server/v1","task_family":"debug_server","agent":"GPT-4o","human_minutes":45.0,"score_binarized":0,"run_id":"run_002"}
```

### Non-Standard Formats (use `convert` or `--column-map`)

If your data uses different column names, you have three options:

**Option A: Auto-detection.** Many common names are auto-detected:
```csv
problem_id,category,model,duration_minutes,correct
```
All of these will be automatically mapped to the standard names.

**Option B: Column mapping.** Use `--column-map`:
```bash
python compute_time_horizon.py compute --input data.csv \
  --column-map "my_model_col=agent,hours_taken=human_hours" --auto-family
```

**Option C: Convert first.** Use the `convert` subcommand:
```bash
python compute_time_horizon.py convert --input raw_data.csv \
  --output standardized.csv --column-map "model=agent" --auto-family --preview
```

### Release Dates Format

```yaml
# release_dates.yaml
date:
  "GPT-4o": "2024-05-13"
  "Claude 3.5 Sonnet": "2024-06-20"
```

---

## 6. Worked Example

### Setup
Consider 2 agents evaluated on 4 tasks across 2 families:

| task_id | task_family | human_minutes | Agent A score | Agent B score |
|---------|-------------|---------------|---------------|---------------|
| t1      | debugging   | 10            | 1, 1          | 1, 0          |
| t2      | debugging   | 60            | 1, 0          | 0, 0          |
| t3      | ml_tasks    | 120           | 0, 0          | 0, 0          |
| t4      | ml_tasks    | 480           | 0, 0          | 0, 0          |

### Step 1: Compute Weights (invsqrt)

Family "debugging" has 2 tasks. Family "ml_tasks" has 2 tasks.

For Agent A, task t1 (2 runs):
- equal component: $1/2 = 0.5$
- family factor: $1/\sqrt{2} \approx 0.707$
- raw weight: $0.5 \times 0.707 = 0.354$

After normalization across all 8 runs for Agent A, each run gets its proportional weight summing to 1.0.

### Step 2: Fit Logistic

For Agent A:
- $X = [\log_2(10), \log_2(10), \log_2(60), \log_2(60), \log_2(120), \log_2(120), \log_2(480), \log_2(480)]$
- $X \approx [3.32, 3.32, 5.91, 5.91, 6.91, 6.91, 8.91, 8.91]$
- $y = [1, 1, 1, 0, 0, 0, 0, 0]$

Fitting `LogisticRegression(C=100000)` yields $\beta_1 < 0$ (negative slope) and some $\beta_0 > 0$.

### Step 3: Extract Horizon

If $\beta_0 = 10.5$ and $\beta_1 = -2.0$:

$$T_{50} = 2^{(0 - 10.5) / (-2.0)} = 2^{5.25} \approx 38 \text{ minutes}$$

### Step 4: Bootstrap

Repeat 1000 times:
1. Resample families: maybe draw [debugging, debugging] (ml_tasks excluded!)
2. Within debugging: resample tasks: maybe draw [t1, t1] (t2 excluded!)
3. Within each (task, agent): resample runs
4. Fit logistic on the resampled data → get $T_{50}^{(b)}$

The 2.5th and 97.5th percentiles of $\{T_{50}^{(b)}\}$ give the 95% CI.

---

## 7. Simplifications & Limitations

This standalone script makes the following simplifications compared to METR's full pipeline:

| Feature | Full Pipeline | This Script |
|---------|--------------|-------------|
| Pipeline orchestration | DVC stages | Direct function calls |
| Benchmark stitching | Merges v1.0 and v1.1 results | Single input file |
| Bootstrap variants | `task_family → task_id → run_id` or with `time_buckets` | `task_family → task_id → run_id` only |
| Task source exclusion | Can exclude SWAA, RE-Bench by source | User pre-filters their data |
| Plotting | 30+ specialized visualizations | Single horizon-vs-date plot |
| Empirical success rates | Computed per time bucket | Not included in output |
| BCE loss diagnostic | Computed per agent | Not included |

**When results may not match the full pipeline:**
1. **Stitched models**: Models from older benchmark versions (e.g., GPT-2, davinci-002) were evaluated on a different task suite and stitched into the final results. This script cannot reproduce stitched results without both datasets.
2. **Doubling time "all_time_stitched"**: The full pipeline computes a stitched all-time doubling time using SOTA models from both benchmark versions. This script only computes the "from_2023_on" trend (which uses only the current benchmark's data).
3. **Bootstrap variance**: CIs depend on the exact bootstrap seed sequence. This script uses the same seeding scheme (`42 + i`) as the original, so CIs should match when given identical input data.
