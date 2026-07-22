import os
import copy
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from cost_fnc_jigs import evaluate_schedule_jigs
from adv_neigh import build_move_context, neighbour_staffing_plan_advanced
from starting_temp import dominates,extract_objectives,pareto_deterioration

# =========================================================
# Pareto utilities
# =========================================================



def update_archive(archive, solution, result):
    """
    Update the Pareto archive with a candidate solution.

    Each archive entry stores:
        {
            "solution": solution_dict,
            "result": evaluation_dict,
            "objectives": np.array([labour_cost, total_tardiness, severity])
        }
    """
    candidate_obj = extract_objectives(result)
    atol = 1e-9

    # Reject if the same objective point is already stored.
    for entry in archive:
        if np.allclose(entry["objectives"], candidate_obj, rtol=0.0, atol=atol):
            return archive

    # Reject if dominated by any archive member
    for entry in archive:
        if dominates(entry["objectives"], candidate_obj):
            return archive

    # Remove archive members dominated by the candidate
    new_archive = [
        entry for entry in archive
        if not dominates(candidate_obj, entry["objectives"])
    ]

    new_archive.append({
        "solution": copy.deepcopy(solution),
        "result": copy.deepcopy(result),
        "objectives": candidate_obj.copy(),
    })

    return new_archive


def attach_station_day_utilization(
    solution,
    result,
    data,
    start_date,
    T_max,
    efficiency_alpha,
):
    """
    Enrich an evaluated result with station-day utilization metrics.

    The archived field is stored as [day, station] so it matches the
    staffing_plan layout.
    """
    move_context = build_move_context(
        staffing_plan=solution["staffing_plan"],
        job_sequence=solution["job_sequence"],
        station_assignment=result["station_assignment"],
        tardiness_vector=result["tardiness_vector"],
        data=data,
        start_date=start_date,
        T_max=T_max,
        efficiency_alpha=efficiency_alpha,
    )

    enriched_result = copy.deepcopy(result)
    enriched_result["station_day_utilization"] = move_context["station_day_utilization"].copy()
    return enriched_result


def archive_lexicographic_best(
    archive,
    tol_tardiness=1e-9,
    tol_cost=1e-9,
):
    """
    Extract the lexicographic best objective triple from the archive.

    Ranking:
    1. minimum tardiness
    2. minimum cost among solutions tied on tardiness
    3. minimum severity among solutions tied on tardiness and cost
    """
    if not archive:
        return None

    archive_objs = np.array([entry["objectives"] for entry in archive], dtype=float)

    best_tardiness = float(np.min(archive_objs[:, 1]))
    tardiness_mask = archive_objs[:, 1] <= best_tardiness + tol_tardiness
    tardiness_slice = archive_objs[tardiness_mask]

    best_cost = float(np.min(tardiness_slice[:, 0]))
    cost_mask = tardiness_mask & (archive_objs[:, 0] <= best_cost + tol_cost)
    cost_slice = archive_objs[cost_mask]

    best_severity = float(np.min(cost_slice[:, 2]))

    return np.array([best_cost, best_tardiness, best_severity], dtype=float)


# =========================================================
# Multiprocessing wrappers
# =========================================================

def generate_neighbour_worker(args):
    """
    Multiprocessing-safe neighbour generator.

    Current neighbourhood:
    - modifies the staffing plan only
    - realized station usage is determined dynamically by the evaluator
    """
    (
        solution,
        move_context,
        start_date,
        max_workers_per_station,
        family_probs,
        redistribution_probs,
        opening_probs,
        closing_probs,
    ) = args

    new_staffing, move_info = neighbour_staffing_plan_advanced(
        staffing_plan=solution["staffing_plan"],
        start_date=start_date,
        max_workers_per_station=max_workers_per_station,
        family_probs=family_probs,
        redistribution_probs=redistribution_probs,
        opening_probs=opening_probs,
        closing_probs=closing_probs,
        move_context=move_context,
    )

    new_solution = {
        "job_sequence": solution["job_sequence"].copy(),
        "staffing_plan": new_staffing,
    }

    if move_info is not None and not move_info.get("recompute_assignment", False):
        new_solution["station_assignment"] = solution["station_assignment"].copy()

    return new_solution, move_info

def evaluate_solution_worker(args):
    """
    Multiprocessing-safe wrapper for schedule evaluation.

    If a neighbour becomes infeasible because the staffing horizon is too short,
    return an infeasible objective vector instead of crashing the SA.
    """
    (
        solution,
        data,
        start_date,
        worker_hourly_cost,
        T_max,
        efficiency_alpha,
        max_workers_per_station,
        n_stations,
    ) = args

    try:
        return evaluate_schedule_jigs(
            job_sequence=solution["job_sequence"],
            staffing_plan=solution["staffing_plan"],
            data=data,
            start_date=start_date,
            worker_hourly_cost=worker_hourly_cost,
            T_max=T_max,
            efficiency_alpha=efficiency_alpha,
            max_workers_per_station=max_workers_per_station,
            n_stations=n_stations,
            station_assignment=solution.get("station_assignment"),
        )

    except ValueError as e:
        if "Staffing horizon is too short" in str(e):
            return {
                "labour_cost": np.inf,
                "severity": np.inf,
                "total_tardiness": np.inf,
                "tardiness_vector": np.array([]),
                "completion_times": np.array([]),
                "assignments": [],
            }
        raise


# =========================================================
# Main Pareto Simulated Annealing
# =========================================================

def SA_pareto(
    data,
    initial_solution,
    start_date,
    worker_hourly_cost,
    T0,
    alpha,
    M,
    N,
    T_max,
    efficiency_alpha,
    max_workers_per_station,
    n_stations,
    objective_scales,
    family_probs=None,
    redistribution_probs=None,
    opening_probs=None,
    closing_probs=None,
    num_cores=None,
    chunk_multiplier=10,
    log_dir=".",
    tardiness_patience=3,
    cost_patience=3,
    severity_patience=3,
    tardiness_tol=1e-9,
    cost_tol=1e-9,
    severity_tol=1e-9,
):
    """
    Parallel Pareto Simulated Annealing for the jigs assembly problem.

    Current decision variables:
    - job_sequence  : fixed ordering in the current version
    - staffing_plan : modified through the advanced neighbourhood

    Acceptance logic:
    - If neighbour dominates current -> accept
    - If current dominates neighbour -> accept with probability exp(-Delta/T)
    - If neither dominates the other -> accept as a Pareto trade-off

    Archive:
    - Stores all non-dominated solutions found during the search

    Parameters
    ----------
    data : pd.DataFrame
        Input data already sorted consistently with the solution.
    initial_solution : dict
        {
            "job_sequence": np.array(...),
            "staffing_plan": np.array(...)
        }
    start_date : datetime-like
    worker_hourly_cost : float
    T0 : float
        Initial temperature.
    alpha : float
        Cooling factor.
    M : int
        Number of outer temperature iterations.
    N : int
        Number of neighbour evaluations per temperature level.
    T_max : float
        Working seconds per day.
    efficiency_alpha : float
        Nonlinear worker efficiency exponent.
    max_workers_per_station : int
    n_stations : int
    objective_scales : array-like length 3
        Scales for [labour_cost, total_tardiness, severity]
        used in normalized deterioration.
    num_cores : int or None
    chunk_multiplier : int
        Parallel chunk size = num_cores * chunk_multiplier.
    log_dir : str
        Folder where the SA log file is written.
    tardiness_patience : int
        Number of outer iterations with no tardiness improvement before cost
        refinement is considered for stopping.
    cost_patience : int
        Number of outer iterations with no cost improvement inside the
        best-tardiness region before severity refinement is considered.
    severity_patience : int
        Number of outer iterations with no severity improvement inside the
        best tardiness + best cost region before stopping.
    tardiness_tol : float
        Tolerance used when comparing tardiness values.
    cost_tol : float
        Tolerance used when comparing cost values.
    severity_tol : float
        Tolerance used when comparing severity values.

    Returns
    -------
    current_solution : dict
        Final current solution.
    current_result : dict
        Evaluation of the final current solution.
    archive : list
        Final Pareto archive.
    filename : str
        Path to the log file.
    """
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    objective_scales = np.asarray(objective_scales, dtype=float)
    chunk_size = max(1, num_cores * chunk_multiplier)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(log_dir, f"Pareto_SA_run_{run_timestamp}.txt")

    # -----------------------------------------------------
    # Initial solution evaluation
    # -----------------------------------------------------
    current_solution = copy.deepcopy(initial_solution)

    current_result = evaluate_schedule_jigs(
        job_sequence=current_solution["job_sequence"],
        staffing_plan=current_solution["staffing_plan"],
        data=data,
        start_date=start_date,
        worker_hourly_cost=worker_hourly_cost,
        T_max=T_max,
        efficiency_alpha=efficiency_alpha,
        max_workers_per_station=max_workers_per_station,
        n_stations=n_stations,
        station_assignment=current_solution.get("station_assignment"),
    )
    current_result = attach_station_day_utilization(
        solution=current_solution,
        result=current_result,
        data=data,
        start_date=start_date,
        T_max=T_max,
        efficiency_alpha=efficiency_alpha,
    )
    current_solution["station_assignment"] = current_result["station_assignment"].copy()
    current_obj = extract_objectives(current_result)

    archive = []
    archive = update_archive(archive, current_solution, current_result)

    T = T0
    total_attempts = 0
    total_accepts = 0
    lex_best_so_far = archive_lexicographic_best(
        archive,
        tol_tardiness=tardiness_tol,
        tol_cost=cost_tol,
    )
    tardiness_plateau_iters = 0
    cost_plateau_iters = 0
    severity_plateau_iters = 0
    early_stopped = False
    early_stop_iter = None
    early_stop_reason = None

    # -----------------------------------------------------
    # Log file
    # -----------------------------------------------------
    with open(filename, "w") as f:
        f.write("===== PARETO SIMULATED ANNEALING RUN =====\n")
        f.write(f"Run Timestamp: {run_timestamp}\n")
        f.write(f"CPU cores: {num_cores}\n")
        f.write(f"Chunk size: {chunk_size}\n\n")

        f.write("---- Hyperparameters ----\n")
        f.write(f"T0: {T0}\n")
        f.write(f"alpha: {alpha}\n")
        f.write(f"M: {M}\n")
        f.write(f"N: {N}\n")
        f.write(f"T_max: {T_max}\n")
        f.write(f"efficiency_alpha: {efficiency_alpha}\n")
        f.write(f"max_workers_per_station: {max_workers_per_station}\n")
        f.write(f"n_stations: {n_stations}\n")
        f.write(f"objective_scales: {objective_scales.tolist()}\n")
        f.write(
            "early_stopping_patience: "
            f"tardiness={tardiness_patience}, cost={cost_patience}, severity={severity_patience}\n"
        )
        f.write(
            "early_stopping_tolerances: "
            f"tardiness={tardiness_tol}, cost={cost_tol}, severity={severity_tol}\n"
        )
        f.write(f"Initial objectives [cost, tardiness, severity]: {current_obj.tolist()}\n")
        f.write("==========================================\n\n")

    # -----------------------------------------------------
    # Main loop
    # -----------------------------------------------------
    executor = ProcessPoolExecutor(max_workers=num_cores)

    try:
        for outer_iter in range(M):
            remaining = N
            iter_attempts = 0
            iter_accepts = 0

            while remaining > 0:
                current_chunk = min(chunk_size, remaining)
                remaining -= current_chunk
                move_context = build_move_context(
                    staffing_plan=current_solution["staffing_plan"],
                    job_sequence=current_solution["job_sequence"],
                    station_assignment=current_result["station_assignment"],
                    tardiness_vector=current_result["tardiness_vector"],
                    data=data,
                    start_date=start_date,
                    T_max=T_max,
                    efficiency_alpha=efficiency_alpha,
                )

                # 1) Generate neighbours in parallel
                neighbour_args = [
                    (
                        current_solution,
                        move_context,
                        start_date,
                        max_workers_per_station,
                        family_probs,
                        redistribution_probs,
                        opening_probs,
                        closing_probs,
                    )
                    for _ in range(current_chunk)
                ]
                neighbour_outputs = list(executor.map(generate_neighbour_worker, neighbour_args))

                neighbour_solutions = [item[0] for item in neighbour_outputs]
                move_infos = [item[1] for item in neighbour_outputs]

                # 2) Evaluate neighbours in parallel
                eval_args = [
                    (
                        sol,
                        data,
                        start_date,
                        worker_hourly_cost,
                        T_max,
                        efficiency_alpha,
                        max_workers_per_station,
                        n_stations,
                    )
                    for sol in neighbour_solutions
                ]
                neighbour_results = list(executor.map(evaluate_solution_worker, eval_args))
                neighbour_objs = [extract_objectives(res) for res in neighbour_results]

                # 3) Acceptance test
                acceptable_indices = []
                for idx, n_obj in enumerate(neighbour_objs):
                    total_attempts += 1
                    iter_attempts += 1

                    # Skip infeasible neighbours returned as [inf, inf, inf]
                    if not np.all(np.isfinite(n_obj)):
                        continue

                    if dominates(n_obj, current_obj):
                        acceptable_indices.append((idx, "dominates", 1.0))

                    elif dominates(current_obj, n_obj):
                        delta = pareto_deterioration(current_obj, n_obj, objective_scales)
                        accept_prob = np.exp(-delta / T) if T > 0 else 0.0

                        if np.random.rand() < accept_prob:
                            acceptable_indices.append((idx, "dominated_accept", accept_prob))

                    else:
                        acceptable_indices.append((idx, "non_dominated_tradeoff", 1.0))

                # 4) Choose one accepted neighbour from the batch
                if acceptable_indices:
                    priority = {
                        "dominates": 0,
                        "non_dominated_tradeoff": 1,
                        "dominated_accept": 2,
                    }

                    chosen_idx, chosen_type, chosen_prob = min(
                        acceptable_indices,
                        key=lambda x: (
                            priority[x[1]],
                            float(neighbour_objs[x[0]][1]),
                            float(neighbour_objs[x[0]][0]),
                            float(neighbour_objs[x[0]][2]),
                        )
                    )

                    current_solution = copy.deepcopy(neighbour_solutions[chosen_idx])
                    current_result = attach_station_day_utilization(
                        solution=current_solution,
                        result=neighbour_results[chosen_idx],
                        data=data,
                        start_date=start_date,
                        T_max=T_max,
                        efficiency_alpha=efficiency_alpha,
                    )
                    current_obj = neighbour_objs[chosen_idx]
                    current_solution["station_assignment"] = current_result["station_assignment"].copy()

                    archive = update_archive(archive, current_solution, current_result)

                    total_accepts += 1
                    iter_accepts += 1

            # Cooling
            acceptance_ratio = iter_accepts / iter_attempts if iter_attempts > 0 else 0.0
            T *= alpha

            archive_objs = np.array([entry["objectives"] for entry in archive], dtype=float)
            best_cost = float(np.min(archive_objs[:, 0]))
            best_tardiness = float(np.min(archive_objs[:, 1]))
            best_severity = float(np.min(archive_objs[:, 2]))
            current_lex_best = archive_lexicographic_best(
                archive,
                tol_tardiness=tardiness_tol,
                tol_cost=cost_tol,
            )

            tardiness_improved = current_lex_best[1] < lex_best_so_far[1] - tardiness_tol
            same_tardiness = abs(current_lex_best[1] - lex_best_so_far[1]) <= tardiness_tol
            cost_improved = (
                same_tardiness
                and current_lex_best[0] < lex_best_so_far[0] - cost_tol
            )
            same_cost = abs(current_lex_best[0] - lex_best_so_far[0]) <= cost_tol
            severity_improved = (
                same_tardiness
                and same_cost
                and current_lex_best[2] < lex_best_so_far[2] - severity_tol
            )

            if tardiness_improved:
                lex_best_so_far = current_lex_best.copy()
                tardiness_plateau_iters = 0
                cost_plateau_iters = 0
                severity_plateau_iters = 0
            else:
                tardiness_plateau_iters += 1

                cost_gate_open = (
                    current_lex_best[1] <= tardiness_tol
                    or tardiness_plateau_iters >= tardiness_patience
                )
                if cost_gate_open:
                    if cost_improved:
                        lex_best_so_far = current_lex_best.copy()
                        cost_plateau_iters = 0
                        severity_plateau_iters = 0
                    else:
                        cost_plateau_iters += 1

                    severity_gate_open = cost_plateau_iters >= cost_patience
                    if severity_gate_open:
                        if severity_improved:
                            lex_best_so_far = current_lex_best.copy()
                            severity_plateau_iters = 0
                        else:
                            severity_plateau_iters += 1

            iteration_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(filename, "a") as f:
                f.write(
                    f"[{iteration_timestamp}] "
                    f"Iter {outer_iter + 1}/{M} | "
                    f"T={T:.6f} | "
                    f"Current=[{current_obj[0]:.4f}, {current_obj[1]:.4f}, {current_obj[2]:.4f}] | "
                    f"ArchiveSize={len(archive)} | "
                    f"BestCost={best_cost:.4f} | "
                    f"BestTardiness={best_tardiness:.4f} | "
                    f"BestSeverity={best_severity:.4f} | "
                    f"LexBest=[{current_lex_best[0]:.4f}, {current_lex_best[1]:.4f}, {current_lex_best[2]:.4f}] | "
                    f"Plateau(T,C,S)=({tardiness_plateau_iters}, {cost_plateau_iters}, {severity_plateau_iters}) | "
                    f"AcceptanceRatio={acceptance_ratio:.4f}\n"
                )

            if severity_plateau_iters >= severity_patience:
                early_stopped = True
                early_stop_iter = outer_iter + 1
                early_stop_reason = (
                    "Early stopping triggered: no lexicographic improvement "
                    f"(tardiness -> cost -> severity) for "
                    f"{tardiness_patience}/{cost_patience}/{severity_patience} patience windows."
                )
                with open(filename, "a") as f:
                    f.write(f"{early_stop_reason}\n")
                break

    finally:
        executor.shutdown()

    # -----------------------------------------------------
    # Final log
    # -----------------------------------------------------
    with open(filename, "a") as f:
        f.write("\n===== FINAL RESULTS =====\n")
        if early_stopped:
            f.write(f"{early_stop_reason} Stopped at Iter {early_stop_iter}/{M}.\n")
        f.write(f"Final current objectives: {current_obj.tolist()}\n")
        f.write(f"Total attempts: {total_attempts}\n")
        f.write(f"Total accepts: {total_accepts}\n")
        f.write(f"Global acceptance ratio: {total_accepts / total_attempts if total_attempts > 0 else 0.0:.4f}\n")
        f.write(f"Final archive size: {len(archive)}\n\n")

        f.write("----- Pareto Archive -----\n")
        for i, entry in enumerate(archive):
            obj = entry["objectives"]
            f.write(
                f"Archive[{i}] -> "
                f"Cost={obj[0]:.6f}, "
                f"Tardiness={obj[1]:.6f}, "
                f"Severity={obj[2]:.6f}\n"
            )

        f.write(f"\nEnd Time: {datetime.now()}\n")
        f.write("==========================================\n")

    return current_solution, current_result, archive, filename
