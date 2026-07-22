import numpy as np
import pandas as pd


def _tardiness_thresholds():
    return np.array(
        [
            6 * 3600,
            12 * 3600,
            24 * 3600,
            3 * 24 * 3600,
            7 * 24 * 3600,
        ],
        dtype=float,
    )


def _tardiness_bin_label(tardiness_sec):
    if pd.isna(tardiness_sec):
        return pd.NA

    thresholds = _tardiness_thresholds()

    if tardiness_sec <= 0:
        return "On Time"
    if tardiness_sec <= thresholds[0]:
        return "0-6h"
    if tardiness_sec <= thresholds[1]:
        return "6-12h"
    if tardiness_sec <= thresholds[2]:
        return "12-24h"
    if tardiness_sec <= thresholds[3]:
        return "1-3 days"
    if tardiness_sec <= thresholds[4]:
        return "3-7 days"
    return ">7 days"


def _format_station_staff(segments):
    unique_workers = list(dict.fromkeys(int(seg["workers"]) for seg in segments))
    if len(unique_workers) == 1:
        return unique_workers[0]
    return ", ".join(str(value) for value in unique_workers)


def builder(
    entry,
    data,
    start_date,
    T_max,
    efficiency_alpha,
):
    """
    Rebuild the production schedule for one selected Pareto archive solution.

    Parameters
    ----------
    entry : dict
        One selected Pareto archive entry, for example ``pareto_archive[4]``.
    data : pd.DataFrame
        Dataset used during evaluation, in the same row order used by the SA
        run. In this project that usually means the deadline-sorted dataframe
        returned by ``jigs_initial_sequence``. It must contain at least
        ``DIMPLE_count`` and, ideally, ``AssemblyRef`` and ``deadline``.
    start_date : datetime-like
        Same start date used during schedule evaluation.
    T_max : float, default 8 * 3600
        Working seconds per day. Use the same value used in the SA run.
    efficiency_alpha : float, default 0.9
        Worker-efficiency exponent. Use the same value used in the SA run.
    Returns
    -------
    pd.DataFrame
        ``fs`` containing the per-segment processing rows already split by day
        during reconstruction.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")

    if "solution" not in entry or "result" not in entry:
        raise ValueError("entry must contain 'solution' and 'result'.")

    solution = entry["solution"]
    result = entry["result"]

    job_sequence = np.asarray(
        solution.get("job_sequence", np.arange(len(data), dtype=int)),
        dtype=int,
    )
    staffing_plan = np.asarray(solution["staffing_plan"], dtype=int)
    station_assignment = np.asarray(
        result.get("station_assignment", solution.get("station_assignment")),
        dtype=int,
    )
    completion_times = np.asarray(result["completion_times"], dtype=float)

    if station_assignment.shape[0] != len(data):
        raise ValueError("station_assignment length must match the number of jobs in data.")

    if completion_times.shape[0] != len(data):
        raise ValueError("completion_times length must match the number of jobs in data.")

    start_date = pd.Timestamp(start_date)
    n_days, n_stations = staffing_plan.shape
    seconds_in_day = 24 * 3600
    station_available_time = np.zeros(n_stations, dtype=float)

    def is_weekend(day_idx):
        return (start_date + pd.Timedelta(days=int(day_idx))).weekday() >= 5

    def next_weekday(day_idx):
        while is_weekend(day_idx):
            day_idx += 1
        return day_idx

    def simulate_processing_segments(station_idx, ready_time, base_work):
        current_time = float(ready_time)
        remaining_work = float(base_work)
        segments = []

        while True:
            day_idx = int(current_time // seconds_in_day)
            time_in_day = current_time % seconds_in_day

            if day_idx >= n_days:
                raise ValueError(
                    "Staffing horizon is too short to rebuild the selected archive solution."
                )

            if is_weekend(day_idx):
                current_time = next_weekday(day_idx) * seconds_in_day
                continue

            workers_today = int(staffing_plan[day_idx, station_idx])

            if workers_today <= 0:
                current_time = (day_idx + 1) * seconds_in_day
                continue

            available_today = T_max - time_in_day
            if available_today <= 0:
                current_time = (day_idx + 1) * seconds_in_day
                continue

            speed_today = workers_today ** efficiency_alpha
            time_needed_today = remaining_work / speed_today
            processing_time_today = min(available_today, time_needed_today)

            segment_start_sec = day_idx * seconds_in_day + time_in_day
            segment_end_sec = segment_start_sec + processing_time_today

            segments.append(
                {
                    "workers": workers_today,
                    "segment_start_sec": float(segment_start_sec),
                    "segment_end_sec": float(segment_end_sec),
                }
            )

            remaining_work -= processing_time_today * speed_today
            if remaining_work <= 1e-9:
                return segments

            current_time = (day_idx + 1) * seconds_in_day

    fs_rows = []

    for job_idx in job_sequence:
        station_idx = int(station_assignment[job_idx])
        base_work = float(data.iloc[job_idx]["DIMPLE_count"]) * 60.0
        segments = simulate_processing_segments(
            station_idx=station_idx,
            ready_time=station_available_time[station_idx],
            base_work=base_work,
        )

        start_sec = float(segments[0]["segment_start_sec"])
        end_sec = float(segments[-1]["segment_end_sec"])
        archived_end_sec = float(completion_times[job_idx])

        if not np.isclose(end_sec, archived_end_sec, rtol=0.0, atol=1e-6):
            raise ValueError(
                "Reconstructed completion times do not match the selected archive entry. "
                "Use the same data, start_date, T_max, and efficiency_alpha as the SA run."
            )

        station_available_time[station_idx] = end_sec

        deadline_value = data.iloc[job_idx]["deadline"] if "deadline" in data.columns else pd.NaT
        if pd.isna(deadline_value):
            if "DeadlineSeconds" in data.columns:
                deadline_value = start_date + pd.to_timedelta(
                    float(data.iloc[job_idx]["DeadlineSeconds"]),
                    unit="s",
                )
            else:
                deadline_value = pd.NaT

        assembly_ref = (
            data.iloc[job_idx]["AssemblyRef"]
            if "AssemblyRef" in data.columns
            else int(job_idx)
        )
        deadline_timestamp = (
            pd.Timestamp(deadline_value) if not pd.isna(deadline_value) else pd.NaT
        )
        product_end_time = start_date + pd.to_timedelta(end_sec, unit="s")

        if pd.isna(deadline_timestamp):
            product_tardiness_sec = np.nan
        else:
            product_tardiness_sec = max(
                (product_end_time - deadline_timestamp).total_seconds(),
                0.0,
            )

        for seg in segments:
            segment_start = start_date + pd.to_timedelta(
                float(seg["segment_start_sec"]),
                unit="s",
            )
            segment_end = start_date + pd.to_timedelta(
                float(seg["segment_end_sec"]),
                unit="s",
            )

            fs_rows.append(
                {
                    "Assembly Reference": assembly_ref,
                    "station": station_idx,
                    "Delay bin": _tardiness_bin_label(product_tardiness_sec),
                    "Workers used": int(seg["workers"]),
                    "Product start time": start_date + pd.to_timedelta(start_sec, unit="s"),
                    "Product end time": product_end_time,
                    "Product tardiness (seconds)": product_tardiness_sec,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "date": segment_start.floor("D"),
                    "processing_time_sec": (
                        segment_end - segment_start
                    ).total_seconds(),
                }
            )

    if fs_rows:
        fs_df = (
            pd.DataFrame(fs_rows)
            .sort_values(["date", "station", "segment_start", "Assembly Reference"])
            .reset_index(drop=True)
        )
    else:
        fs_df = pd.DataFrame(
            columns=[
                "Assembly Reference",
                "station",
                "Delay bin",
                "Workers used",
                "Product start time",
                "Product end time",
                "Product tardiness (seconds)",
                "segment_start",
                "segment_end",
                "date",
                "processing_time_sec",
            ]
        )

    return fs_df
