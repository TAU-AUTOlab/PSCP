# PSCP Experiment Package

VNS solver for the Partial Set Covering Problem.

---

## Step 1 — Compile the solver (do this once)

**Linux / university server:**
```bash
bash compile.sh
```

**Windows:**
```
compile.bat
```

This produces `PSCP_solver_V02` (Linux) or `PSCP_solver_V02.exe` (Windows) in the package folder.

---

## Step 2 — Run experiments

### Guided menu (recommended)

Just run:
```bash
python run_experiments.py
```

The program will ask you step by step:

1. **What type?** Choose one:
   - `Baseline` — run the algorithm with its default settings
   - `Sensitivity` — test how changing one parameter affects results
   - `Ablation` — test what happens when one search move is removed
   - `Everything` — run all 24 configurations

2. **Which parameter / move?** A numbered list appears — enter the number(s) you want, comma-separated, or type `all`.

3. **How many repetitions?** Each rep uses a different random seed. Enter a number (default: 10).

4. **Instances?** Choose all 21 (full study) or 3 small ones (quick check).

5. **Coverage targets?** 90%, 95%, or both.

6. **Time per run (seconds)?** Default is 100s.

The program then shows a summary of how many runs will be launched and how long it will take, and asks you to confirm before starting.

---

### Direct launch (for running on a server without interaction)

```bash
python run_experiments.py --exp <what> --inst all --covs 90,95 --reps 10 --time 100
```

`--exp` accepts group codes, specific IDs, or a comma-separated mix:

| Code | Group | Configs | Values tested (baseline in parentheses) |
|---|---|:---:|---|
| `0` | Baseline | 1 | all defaults |
| `1` | Sensitivity xi | 4 | 0.03, 0.05, 0.09, 0.11  (baseline: 0.07) |
| `2` | Sensitivity STAG | 4 | 200, 300, 500, 600  (baseline: 400) |
| `3` | Sensitivity RL_const | 4 | 4, 6, 10, 12  (baseline: 8) |
| `4` | Sensitivity RL_imp | 4 | 90, 110, 150, 170  (baseline: 128) |
| `5` | Sensitivity t_ls | 4 | 150ms, 200ms, 300ms, 350ms  (baseline: 250ms) |
| `6` | Ablation | 3 | no drop moves / no swap moves / no N2 moves |
| `A` | **All** | **24** | 1 + 5x4 + 3 = 24 total configurations |

To run specific individual configs by ID instead of a whole group: `--exp xi_003,stag_600`

**Quick smoke test (verify the setup works, takes ~5 minutes):**
```bash
python run_experiments.py --smoke
```

---

## Splitting across multiple servers

The cleanest way to use multiple servers is to give each server the **same full experiment set** (`--exp A --inst all --covs 90,95`) but a **different slice of repetitions** using `--rep-start` and `--rep-end`.

Every server runs all 24 configurations on all 21 instances — just for fewer reps each.
Results merge together automatically (each rep has a unique number across servers).
Each server takes roughly equal time.

Each server gets the **same experiment set and rep count** — you just tell it which server number it is. The script splits the reps automatically.

| Flag | What it controls | Example |
|---|---|---|
| `--reps 10` | Total repetitions across all servers | 10 independent runs per configuration |
| `--server N` | This server's index (1, 2, 3, …) | `--server 2` |
| `--of M` | Total number of servers | `--of 3` |
| `--time 100` | Seconds the solver runs per instance | change to 60 or 200 as needed |

> `--exp A` means **all 24 configurations** (1 baseline + 5 parameters x 4 values + 3 ablation variants). Replace `A` with a group code from the table above to run a subset, e.g. `--exp 6` for ablation only.

### 3 servers — 10 reps total, split automatically (~42h each)

```bash
# Server 1
python run_experiments.py --exp A --inst all --covs 90,95 --reps 10 --server 1 --of 3 --time 100

# Server 2
python run_experiments.py --exp A --inst all --covs 90,95 --reps 10 --server 2 --of 3 --time 100

# Server 3
python run_experiments.py --exp A --inst all --covs 90,95 --reps 10 --server 3 --of 3 --time 100
```

### 4 servers — 10 reps total, split automatically (~35h each)

```bash
# Server 1
python run_experiments.py --exp A --inst all --covs 90,95 --reps 10 --server 1 --of 4 --time 100

# Server 2
python run_experiments.py --exp A --inst all --covs 90,95 --reps 10 --server 2 --of 4 --time 100

# Server 3
python run_experiments.py --exp A --inst all --covs 90,95 --reps 10 --server 3 --of 4 --time 100

# Server 4
python run_experiments.py --exp A --inst all --covs 90,95 --reps 10 --server 4 --of 4 --time 100
```

The script prints which rep range it assigned to each server (e.g. `reps 4-7  (4 of 10 total, server 2 of 3)`) so you can verify before it starts.

Each server writes its own timestamped CSV to `results/`. After all servers finish, copy all CSV files into a single `results/` folder and run `progress.py` or merge them (see Step 4 below).

---

## Step 3 — Check progress during a run

From a separate terminal, at any time:
```bash
python progress.py
```

This reads all result files and shows how many runs are done, how many remain, and the estimated time to finish. Pass the same flags you gave to `run_experiments.py` if you are running a subset:
```bash
python progress.py --exp 0,6 --inst all --covs 90,95 --reps 10 --time 100
```

---

## Step 4 — Collect results

Each session writes one file: `results/results_YYYYMMDD_HHMMSS.csv`

If you ran multiple sessions or servers in parallel, merge all CSVs:
```bash
# Linux
head -1 results/$(ls results/ | head -1) > results/all_results.csv
tail -n +2 -q results/results_*.csv >> results/all_results.csv
```

Each row in the CSV contains: `exp_id`, `instance`, `cov_pct`, `rep`, `seed`, `best_obj`, `rt_s`, `wall_s`, `selected_set` (a space-separated list of the 1-based indices of the selected suppliers), and all parameter settings used.

---

## Running on a SLURM cluster

Save as `submit.sh`, then `sbatch submit.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=PSCP
#SBATCH --array=0-6
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%A_%a.out

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
python run_experiments.py --exp "$SLURM_ARRAY_TASK_ID" --inst all --covs 90,95 --reps 10 --time 100
```

---

## Requirements

- Python 3.6+
- g++ with C++17 support (Linux) or MinGW/MSVC (Windows)
