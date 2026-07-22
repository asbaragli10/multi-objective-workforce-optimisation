# Multi-objective workforce optimisation for off-site assembly

This repository implements the parallel Pareto Simulated Annealing framework presented in **“Multi-objective Workforce Allocation for Off-site Construction Assembly Using Parallel Simulated Annealing”** (TCOT 2026).

The method creates a day-by-day staffing plan for parallel assembly stations. It balances delivery reliability against labour expenditure while accounting for diminishing productivity gains when several workers share one station.

## Optimisation problem

![Assembly off-site construction job shop](Problem.jpg)

*Assembly job-shop formulation from Figure 1 of the accompanying TCOT 2026 paper.*

Products enter the system in a fixed **Earliest Due Date (EDD)** sequence. For every product, the scheduler selects the station that gives the earliest feasible completion time under the current staffing plan and station availability.

The core decision variable is the staffing matrix

\[
W = [w_{d,s}] \in \mathbb{Z}^{D \times S},
\]

where \(w_{d,s}\) is the number of workers assigned to station \(s\) on day \(d\). A value of zero closes that station for the day. Each station is limited to `max_workers_per_station`, and its effective processing rate is

\[
v_{d,s} = w_{d,s}^{\alpha}, \qquad 0 < \alpha \leq 1,
\]

so the efficiency exponent \(\alpha\) captures coordination and workspace losses when multiple workers collaborate.

The optimiser minimizes three objectives:

1. **Labour cost** - staffed worker-hours up to the last production day, multiplied by the hourly labour rate.
2. **Total tardiness** - the sum of `max(0, completion time - deadline)` across all jobs.
3. **Delay severity** - the mean tardiness among delayed jobs only.

Weekends are treated as non-working days. Jobs may continue across multiple working days, but jobs assigned to the same station cannot overlap.

## Context-aware neighbourhood moves

The modular neighbourhood implementation is in [`adv_neigh.py`](adv_neigh.py). `build_move_context()` reconstructs four signals from the current solution:

- **Tardiness pressure** identifies days contributing to late completion and propagates part of that pressure to preceding working days.
- **Slack** measures the positive margin between completion and deadline for on-time jobs.
- **Station-day utilisation** shows where available staffed capacity is being consumed.
- **Last used day** limits edits to the effective production horizon plus a short look-ahead buffer.

`neighbour_staffing_plan_advanced()` samples one of three move families and retries when the sampled move is infeasible.

| Family | Operator | Within-family probability | Staffing change | Station assignment |
|---|---|---:|---|---|
| Redistribution | `rebalance_day_active` | 38% | Transfers one worker between two active stations on the same day. | Reused |
| Redistribution | `shift_same_station` | 30% | Moves one worker for the same station from a slack day to a higher-pressure day. | Reused |
| Redistribution | `single_cell_active` | 22% | Increases or decreases one active station-day by one worker. | Reused |
| Redistribution | `station_block_active` | 10% | Increases or decreases one active station over a consecutive 2-3 day block. | Reused |
| Opening | `open_station_day` | 35% | Activates an unused station on one high-pressure day. | Recomputed |
| Opening | `open_station_block` | 65% | Activates an unused station over a consecutive 2-3 day block. | Recomputed |
| Closing | `close_station_day` | 85% | Deactivates a low-risk station-day while keeping another station active. | Recomputed |
| Closing | `close_station_block` | 15% | Deactivates a station over a low-pressure 2-3 day block. | Recomputed |

Redistribution moves preserve the active station structure and therefore reuse the realised job-to-station assignment. Opening and closing moves change station availability, so dynamic dispatch is recomputed. This is a deliberate runtime trade-off described in the paper.

The family probabilities are configurable independently of the within-family probabilities. The defaults in `adv_neigh.py` are 70% redistribution, 15% opening, and 15% closing. The experiment configured in `main.py` uses 80%, 20%, and 0%, respectively.

## Parallel Pareto Simulated Annealing

[`MT_SA.py`](MT_SA.py) generates and evaluates candidate staffing plans in parallel batches. Candidate acceptance follows these rules:

- a neighbour that Pareto-dominates the current solution is accepted;
- a neighbour dominated by the current solution may be accepted according to its temperature-scaled deterioration;
- a mutually non-dominated neighbour is accepted as a valid trade-off.

When several candidates in a batch are acceptable, the search prioritizes dominating solutions, then non-dominated trade-offs, then probabilistically accepted dominated solutions. Ties are resolved by tardiness, labour cost, and severity. An external archive retains unique non-dominated solutions.

## Dataset interface

The industrial case-study data are **not distributed in this public repository**. Supply a local CSV with the following required fields:

| Column | Type | Description |
|---|---|---|
| `AssemblyRef` | string | Product or assembly identifier. |
| `DIMPLE_count` | integer | Assembly workload proxy; the evaluator currently uses 60 seconds of base work per unit. |
| `deadline` | date | Latest acceptable completion date, parsed day-first. |

Additional production metadata can remain in the file but are not used by the optimiser. The case-study preparation workflow also retained `Len`, `Coil`, and `Howick Work Time`.

The following rows are **illustrative synthetic examples**, not records from the industrial dataset:

| AssemblyRef | Len | Coil | Howick Work Time | DIMPLE_count | deadline |
|---|---:|---:|---:|---:|---|
| FRAME-001 | 17884.80 | 100 | 214.42 | 30 | 12-May-25 |
| FRAME-002 | 23396.85 | 100 | 291.12 | 52 | 12-May-25 |
| FRAME-003 | 41170.61 | 100 | 508.71 | 82 | 19-May-25 |
| FRAME-004 | 17626.18 | 100 | 284.25 | 58 | 10-Jun-25 |
| FRAME-005 | 6557.92 | 100 | 112.76 | 20 | 02-Jul-25 |

Before running the current entry point, save your local input as `Berwick_full_deadlines.csv` in the repository root or update the `pd.read_csv(...)` path near the top of `main.py`. The filename is ignored by Git to prevent accidental publication.

## Repository structure

| Path | Purpose |
|---|---|
| `main.py` | Configures the case study, estimates the initial temperature, runs the search, and writes results. |
| `initial_solution.py` | Builds the EDD job sequence and initial staffing horizon. |
| `cost_fnc_jigs.py` | Simulates dynamic dispatch and calculates the three objectives. |
| `starting_temp.py` | Calibrates the initial SA temperature from sampled neighbours. |
| `adv_neigh.py` | Builds move context and implements the modular neighbourhood operators. |
| `MT_SA.py` | Runs parallel Pareto SA and maintains the non-dominated archive. |
| `Evaluation/` | Reconstructs selected schedules and produces time/station statistics. |
| `Plots/` | Contains KPI plotting notebooks and publication figures. |

`custom_search.py` and `neigh.py` contain earlier neighbourhood prototypes; the active search uses `adv_neigh.py`.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the optimiser

1. Prepare the input CSV described above.
2. Review the problem and SA parameters in `main.py`.
3. Run from the repository root:

```bash
python main.py
```

The default configuration writes:

- `initial_solution.pkl` and `initial_output.pkl`;
- a timestamped `Pareto_SA_run_*.txt` log;
- `pareto_archive.pkl` containing complete non-dominated solutions;
- `pareto_frontier_summary.csv` containing objective values only.

The full case study is computationally intensive because each candidate requires a complete schedule simulation. Adjust `M`, `N`, `num_cores`, and `chunk_multiplier` in `main.py` for exploratory runs.

## Analysing results

The notebooks in `Evaluation/` load a saved solution or Pareto archive, reconstruct product-level processing segments, and calculate delay, cost, makespan, and station-utilisation statistics. `Plots/production_KPIs.ipynb` creates the KPI figures from the aggregate workbooks in `Plots/`.

## Citation

The accompanying paper has been accepted for TCOT 2026. The full citation will be provided in due course once the conference proceedings are published.
