import numpy as np
import pandas as pd


def evaluate_schedule_jigs(
    job_sequence,
    staffing_plan,
    data,
    start_date,
    worker_hourly_cost,
    T_max,
    efficiency_alpha,
    max_workers_per_station,
    n_stations,
    station_assignment=None,
):
    """
    Evaluate a schedule under dynamic dispatch.

    Jobs are considered in the order given by `job_sequence`.

    Two evaluation modes are supported:
    - Dynamic dispatch: if `station_assignment` is None, each job is assigned
      to the station that yields the earliest completion time under the current
      staffing plan.
    - Fixed assignment reuse: if `station_assignment` is provided, the
      evaluator keeps that realized assignment and only recomputes timing under
      the new staffing plan.
    """
    job_sequence = np.asarray(job_sequence, dtype=int)
    staffing_plan = np.asarray(staffing_plan, dtype=int)
    start_date = pd.Timestamp(start_date)

    n_jobs = len(data)

    if len(job_sequence) != n_jobs:
        raise ValueError("job_sequence length must match number of jobs")

    if set(job_sequence.tolist()) != set(range(n_jobs)):
        raise ValueError("job_sequence must be a permutation of job indices")

    if station_assignment is not None:
        station_assignment = np.asarray(station_assignment, dtype=int)
        if len(station_assignment) != n_jobs:
            raise ValueError("station_assignment length must match number of jobs")
        if np.any(station_assignment < 0) or np.any(station_assignment >= n_stations):
            raise ValueError(f"station_assignment entries must be in [0, {n_stations-1}]")

    if staffing_plan.ndim != 2 or staffing_plan.shape[1] != n_stations:
        raise ValueError(f"staffing_plan must have shape (n_days, {n_stations})")

    if np.any(staffing_plan < 0) or np.any(staffing_plan > max_workers_per_station):
        raise ValueError(f"Each staffing entry must be between 0 and {max_workers_per_station}")

    if not (0 < efficiency_alpha <= 1):
        raise ValueError("efficiency_alpha must be in (0, 1]")

    n_days = staffing_plan.shape[0]
    SECONDS_IN_DAY = 24 * 3600

    tardiness = np.zeros(n_jobs, dtype=float)
    completion_times = np.zeros(n_jobs, dtype=float)
    chosen_station = np.full(n_jobs, -1, dtype=int)
    assignments = []

    station_available_time = np.zeros(n_stations, dtype=float)

    def is_weekend(day_idx):
        return (start_date + pd.Timedelta(days=int(day_idx))).weekday() >= 5

    def next_weekday(day_idx):
        while is_weekend(day_idx):
            day_idx += 1
        return day_idx

    def simulate_completion_on_station(station_idx, ready_time, base_work):
        test_time = ready_time
        remaining_work = base_work

        while True:
            day_idx = int(test_time // SECONDS_IN_DAY)
            time_in_day = test_time % SECONDS_IN_DAY

            if day_idx >= n_days:
                return np.inf

            if is_weekend(day_idx):
                day_idx = next_weekday(day_idx)
                test_time = day_idx * SECONDS_IN_DAY
                continue

            workers_today = staffing_plan[day_idx, station_idx]

            if workers_today <= 0:
                test_time = (day_idx + 1) * SECONDS_IN_DAY
                continue

            available_today = T_max - time_in_day
            if available_today <= 0:
                test_time = (day_idx + 1) * SECONDS_IN_DAY
                continue

            speed_today = workers_today ** efficiency_alpha
            work_done_today = available_today * speed_today

            if work_done_today >= remaining_work:
                time_needed_today = remaining_work / speed_today
                return day_idx * SECONDS_IN_DAY + time_in_day + time_needed_today

            remaining_work -= work_done_today
            test_time = (day_idx + 1) * SECONDS_IN_DAY

    dynamic_dispatch = station_assignment is None

    for job_idx in job_sequence:
        dimple_count = float(data.iloc[job_idx]["DIMPLE_count"])
        deadline = float(data.iloc[job_idx]["DeadlineSeconds"])
        base_work = dimple_count * 60.0

        if dynamic_dispatch:
            best_station = None
            best_completion = None

            for station_idx in range(n_stations):
                completion = simulate_completion_on_station(
                    station_idx=station_idx,
                    ready_time=station_available_time[station_idx],
                    base_work=base_work,
                )

                if best_completion is None or completion < best_completion:
                    best_completion = completion
                    best_station = station_idx
        else:
            best_station = int(station_assignment[job_idx])
            best_completion = simulate_completion_on_station(
                station_idx=best_station,
                ready_time=station_available_time[best_station],
                base_work=base_work,
            )

        if best_station is None or not np.isfinite(best_completion):
            raise ValueError(
                "Staffing horizon is too short: at least one job cannot be completed "
                "within the provided staffing_plan."
            )

        station_available_time[best_station] = best_completion
        chosen_station[job_idx] = best_station
        completion_times[job_idx] = best_completion
        tardiness[job_idx] = max(0.0, best_completion - deadline)

        assignments.append({
            "job_index": int(job_idx),
            "assembly_ref": data.iloc[job_idx]["AssemblyRef"] if "AssemblyRef" in data.columns else int(job_idx),
            "station": int(best_station),
            "completion_time_sec": float(best_completion),
            "tardiness_sec": float(tardiness[job_idx]),
        })

    positive_delay = tardiness[tardiness > 0]
    if positive_delay.size > 0:
        severity = float(positive_delay.mean())
    else:
        severity = 0.0

    last_completion = completion_times.max() if n_jobs > 0 else 0.0
    last_used_day = int(np.floor(last_completion / SECONDS_IN_DAY))

    effective_staffing = staffing_plan[:last_used_day + 1, :]

    weekday_mask = np.array(
        [(start_date + pd.Timedelta(days=d)).weekday() < 5 for d in range(effective_staffing.shape[0])],
        dtype=bool,
    )

    effective_staffing = effective_staffing[weekday_mask, :]
    total_staffed_hours = effective_staffing.sum() * (T_max / 3600.0)
    labour_cost = total_staffed_hours * worker_hourly_cost

    total_tardiness = tardiness.sum()
    makespan = last_completion

    return {
        "labour_cost": labour_cost,
        "severity": severity,
        "total_tardiness": total_tardiness,
        "makespan": makespan,
        "tardiness_vector": tardiness,
        "completion_times": completion_times,
        "station_assignment": chosen_station,
        "assignments": assignments,
    }
