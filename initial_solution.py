import numpy as np
import pandas as pd

# The initial solution defines my station assignment rule:
# 1- Initial job order, built according to Earliest Due Date.
# 2 - Initial staffing plan with shape (n_days, n_stations)


def jigs_initial_sequence(
    data,
    start_date,
    efficiency_alpha,
    T_max,
    n_stations,
    active_stations,
    workers_per_active_station,
    horizon_buffer_days=20,
):
    """
    Initial solution for the dynamic-dispatch formulation.

    Outputs
    -------
    job_sequence : np.ndarray
        Initial job order, built according to Earliest Due Date.
    staffing_plan : np.ndarray
        Initial staffing plan with shape (n_days, n_stations).
    data_sorted : pd.DataFrame
        Data sorted consistently with the returned job sequence.

    Notes
    -----
    This initializer does not create a fixed station assignment. The effective
    station usage is intended to be generated later by the schedule evaluator
    from the job sequence and the staffing plan.
    """
    start_date = pd.Timestamp(start_date)

    data_sorted = data.sort_values("deadline").reset_index(drop=True).copy()

    if "DeadlineSeconds" not in data_sorted.columns:
        deadline_ts = pd.to_datetime(data_sorted["deadline"], dayfirst=True)
        data_sorted["DeadlineSeconds"] = (deadline_ts - start_date).dt.total_seconds()

    if not (1 <= active_stations <= n_stations):
        raise ValueError("active_stations must be between 1 and n_stations")

    if workers_per_active_station < 0:
        raise ValueError("workers_per_active_station must be non-negative")

    active_station_ids = np.arange(active_stations)

    # Initial dispatching rule: Earliest Due Date.
    job_sequence = np.arange(len(data_sorted), dtype=int)

    total_base_work = float(data_sorted["DIMPLE_count"].astype(float).sum()) * 60.0
    station_speed = workers_per_active_station ** efficiency_alpha
    daily_parallel_capacity = active_stations * station_speed * T_max

    if daily_parallel_capacity <= 0:
        required_working_days = 1
    else:
        required_working_days = int(np.ceil(total_base_work / daily_parallel_capacity))

    # Convert required working days into calendar days, since the evaluator
    # skips weekends and the staffing plan is indexed in calendar days.
    working_days_needed = max(1, required_working_days + horizon_buffer_days)
    current_day = 0
    counted_working_days = 0
    while counted_working_days < working_days_needed:
        if (start_date + pd.Timedelta(days=current_day)).weekday() < 5:
            counted_working_days += 1
        current_day += 1

    n_days = current_day

    staffing_plan = np.zeros((n_days, n_stations), dtype=int)
    staffing_plan[:, active_station_ids] = workers_per_active_station

    return job_sequence, staffing_plan, data_sorted
