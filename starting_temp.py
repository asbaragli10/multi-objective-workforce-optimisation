import numpy as np
import copy


def dominates(obj_a, obj_b):
    """
    Pareto dominance test for minimization.

    obj_a dominates obj_b if:
    - obj_a is no worse in all objectives
    - obj_a is strictly better in at least one objective
    """
    a = np.asarray(obj_a, dtype=float)
    b = np.asarray(obj_b, dtype=float)

    return np.all(a <= b) and np.any(a < b)


def extract_objectives(result):
    """
    Convert the schedule evaluation output into a 3-objective vector:
    [labour_cost, total_tardiness, severity]
    """
    return np.array([
        result["labour_cost"],
        result["total_tardiness"],
        result["severity"],
    ], dtype=float)


def pareto_deterioration(current_obj, neighbour_obj, scales):
    """
    Normalized deterioration used only for SA temperature / acceptance.

    It sums only the worsened parts of the move:
        Delta = sum_k max(0, (f_k(neighbour) - f_k(current)) / scale_k)

    This is NOT a weighted-sum objective.
    It is only a way to measure how much worse a move is in multi-objective space.
    """
    current_obj = np.asarray(current_obj, dtype=float)
    neighbour_obj = np.asarray(neighbour_obj, dtype=float)
    scales = np.asarray(scales, dtype=float)

    return np.sum(np.maximum(0.0, (neighbour_obj - current_obj) / scales))


def estimate_T0_pareto(
    initial_solution,
    evaluate_solution,
    neighbour_function,
    n_samples=200,
    target_acceptance=0.8,
    scale_mode="median_abs_change",
):
    """
    Estimate the initial temperature T0 for Pareto Simulated Annealing.

    The method:
    1. Samples random neighbours around the initial solution
    2. Evaluates the 3-objective vectors:
           [labour_cost, total_tardiness, severity]
    3. Builds objective scales
    4. Computes a normalized deterioration Delta for sampled neighbours
    5. Sets T0 so that a typical worsening move is accepted with probability target_acceptance

    Parameters
    ----------
    initial_solution : dict
        Example:
            {
                "station_assignment": ...,
                "staffing_plan": ...
            }

    evaluate_solution : callable
        Function that takes a solution and returns a dict with:
            labour_cost, total_tardiness, severity

    neighbour_function : callable
        Function that takes a solution and returns:
            new_solution, move_info
        or just:
            new_solution

    n_samples : int
        Number of random neighbour samples.

    target_acceptance : float
        Desired initial acceptance probability for a typical worse move.
        Common choices: 0.7 to 0.9

    scale_mode : str
        How to estimate objective scales.
        Options:
        - "median_abs_change" : robust local scale from sampled neighbour changes
        - "initial_value"     : uses objective values at the initial solution

    Returns
    -------
    T0 : float
        Estimated initial temperature.

    details : dict
        Extra information for debugging:
        - current_obj
        - scales
        - positive_deltas
        - sampled_objectives
        - median_positive_delta
    """

    if not (0 < target_acceptance < 1):
        raise ValueError("target_acceptance must be between 0 and 1")

    current_result = evaluate_solution(initial_solution)
    current_obj = extract_objectives(current_result)

    sampled_objectives = []
    sampled_abs_changes = []
    positive_deltas = []

    for _ in range(n_samples):
        neighbour_output = neighbour_function(initial_solution)

        if isinstance(neighbour_output, tuple):
            neighbour_solution = neighbour_output[0]
        else:
            neighbour_solution = neighbour_output

        neighbour_result = evaluate_solution(neighbour_solution)
        neighbour_obj = extract_objectives(neighbour_result)

        sampled_objectives.append(neighbour_obj)
        sampled_abs_changes.append(np.abs(neighbour_obj - current_obj))

    sampled_objectives = np.asarray(sampled_objectives, dtype=float)
    sampled_abs_changes = np.asarray(sampled_abs_changes, dtype=float)

    if scale_mode == "median_abs_change":
        scales = np.median(sampled_abs_changes, axis=0)
        scales = np.where(scales <= 1e-12, 1.0, scales)

    elif scale_mode == "initial_value":
        scales = np.abs(current_obj)
        scales = np.where(scales <= 1e-12, 1.0, scales)

    else:
        raise ValueError("scale_mode must be 'median_abs_change' or 'initial_value'")

    for neighbour_obj in sampled_objectives:
        delta = pareto_deterioration(current_obj, neighbour_obj, scales)
        if delta > 0:
            positive_deltas.append(delta)

    positive_deltas = np.asarray(positive_deltas, dtype=float)

    if len(positive_deltas) == 0:
        raise ValueError(
            "No positive deteriorations found. "
            "The neighbourhood may be too weak, too improving, or the scales may be degenerate."
        )

    typical_delta = np.median(positive_deltas)
    T0 = -typical_delta / np.log(target_acceptance)

    details = {
        "current_obj": current_obj,
        "scales": scales,
        "positive_deltas": positive_deltas,
        "sampled_objectives": sampled_objectives,
        "median_positive_delta": typical_delta,
    }

    return T0, details



