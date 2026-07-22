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


def time_stats(
    fs,
    staffing_plan,
    start_date,
    worker_hourly_cost,
    tardiness_vector,
    T_max=8 * 3600,
):
    """
    Build the time-based outputs used in the analysis:

    1. day-level schedule statistics
    2. aggregate delay distribution

    Parameters
    ----------
    fs : pd.DataFrame
        Per-segment dataframe returned by ``reconstructing.builder``.
    staffing_plan : array-like
        Staffing plan with shape ``(n_days, n_stations)``.
    start_date : datetime-like
        Same schedule start date used during evaluation.
    worker_hourly_cost : float
        Hourly cost per worker.
    tardiness_vector : array-like
        Tardiness values used to build the aggregate delay distribution.
    T_max : float, default 8 * 3600
        Working seconds per day.
    """
    start = pd.Timestamp(start_date)
    SECONDS_IN_DAY = 24 * 3600

    if not isinstance(fs, pd.DataFrame):
        raise ValueError("fs must be a pandas DataFrame.")

    required_columns = [
        "station",
        "Delay bin",
        "Product start time",
        "Product end time",
        "Product tardiness (seconds)",
        "segment_end",
    ]
    missing_columns = [col for col in required_columns if col not in fs.columns]
    if missing_columns:
        raise ValueError("fs is missing required columns: " + ", ".join(missing_columns))

    staffing_plan = np.asarray(staffing_plan, dtype=int)
    if staffing_plan.ndim != 2:
        raise ValueError("staffing_plan must be a 2D array with shape (n_days, n_stations).")

    fs = fs.copy()
    fs["Product start time"] = pd.to_datetime(fs["Product start time"])
    fs["Product end time"] = pd.to_datetime(fs["Product end time"])
    fs["segment_end"] = pd.to_datetime(fs["segment_end"])
    fs["Product tardiness (seconds)"] = pd.to_numeric(
        fs["Product tardiness (seconds)"],
        errors="coerce",
    )
    fs["station"] = fs["station"].astype(int)

    schedule = fs[fs["segment_end"] == fs["Product end time"]].copy()
    schedule["Station Assigned"] = schedule["station"]

    n_days = staffing_plan.shape[0]
    table = []

    for d in range(n_days):
        date = start + pd.Timedelta(days=d)

        if date.weekday() >= 5:
            continue

        day_start = d * SECONDS_IN_DAY
        day_end = (d + 1) * SECONDS_IN_DAY
        workday_start = date
        workday_end = date + pd.Timedelta(seconds=T_max)

        completed_today = schedule[
            (schedule["Product end time"] >= workday_start)
            & (schedule["Product end time"] < workday_end)
        ]
        jobs_processed = int(len(completed_today))

        workers_station = staffing_plan[d].astype(int)
        total_workers = int(workers_station.sum())
        active_stations = int(np.count_nonzero(workers_station))

        labour_cost_day = total_workers * (T_max / 3600.0) * worker_hourly_cost

        tardiness_day_values = completed_today["Product tardiness (seconds)"].dropna()
        tardiness_day = float(tardiness_day_values.sum()) if not tardiness_day_values.empty else 0.0
        positive_tardiness_day = tardiness_day_values[tardiness_day_values > 0]
        severity_day = (
            float(positive_tardiness_day.mean())
            if not positive_tardiness_day.empty
            else 0.0
        )

        if jobs_processed > 0:
            delay_bins = completed_today["Delay bin"].fillna("Unknown")
            on_time = int((delay_bins == "On Time").sum())
            bin_0_6h = int((delay_bins == "0-6h").sum())
            bin_6_12h = int((delay_bins == "6-12h").sum())
            bin_12_24h = int((delay_bins == "12-24h").sum())
            bin_1_3d = int((delay_bins == "1-3 days").sum())
            bin_3_7d = int((delay_bins == "3-7 days").sum())
            bin_gt_7d = int((delay_bins == ">7 days").sum())

            pct_on_time = 100.0 * on_time / jobs_processed
            pct_0_6h = 100.0 * bin_0_6h / jobs_processed
            pct_6_12h = 100.0 * bin_6_12h / jobs_processed
            pct_12_24h = 100.0 * bin_12_24h / jobs_processed
            pct_1_3d = 100.0 * bin_1_3d / jobs_processed
            pct_3_7d = 100.0 * bin_3_7d / jobs_processed
            pct_gt_7d = 100.0 * bin_gt_7d / jobs_processed
        else:
            pct_on_time = 0.0
            pct_0_6h = 0.0
            pct_6_12h = 0.0
            pct_12_24h = 0.0
            pct_1_3d = 0.0
            pct_3_7d = 0.0
            pct_gt_7d = 0.0

        row = {
            "day": d,
            "date": date.date(),
            "workers_total": total_workers,
            "stations_active": active_stations,
            "jobs_processed": jobs_processed,
            "labour_cost_day": labour_cost_day,
            "tardiness_day_sec": tardiness_day,
            "tardiness_day_hr": tardiness_day / 3600.0,
            "severity_day": severity_day,
            "On Time %": pct_on_time,
            "0-6h %": pct_0_6h,
            "6-12h %": pct_6_12h,
            "12-24h %": pct_12_24h,
            "1-3 day %": pct_1_3d,
            "3-7 days %": pct_3_7d,
            "> 7 days %": pct_gt_7d,
        }

        table.append(row)

    schedule_stats_df = pd.DataFrame(table)

    if not schedule_stats_df.empty:
        schedule_stats_df = schedule_stats_df[
            schedule_stats_df["jobs_processed"] != 0
        ].reset_index(drop=True)

    thresholds = _tardiness_thresholds()

    tardiness = np.asarray(tardiness_vector)
    n = len(tardiness)

    bins = {
        "On Time": np.sum(tardiness == 0),
        "0-6h": np.sum((tardiness > 0) & (tardiness <= thresholds[0])),
        "6-12h": np.sum((tardiness > thresholds[0]) & (tardiness <= thresholds[1])),
        "12-24h": np.sum((tardiness > thresholds[1]) & (tardiness <= thresholds[2])),
        "1-3 days": np.sum((tardiness > thresholds[2]) & (tardiness <= thresholds[3])),
        "3-7 days": np.sum((tardiness > thresholds[3]) & (tardiness <= thresholds[4])),
        ">7 days": np.sum(tardiness > thresholds[4]),
    }

    delay_distribution_df = pd.DataFrame({
        "bin": bins.keys(),
        "count": bins.values(),
    })

    if n > 0:
        delay_distribution_df["percentage"] = 100 * delay_distribution_df["count"] / n
    else:
        delay_distribution_df["percentage"] = 0.0

    return schedule_stats_df, delay_distribution_df






def split_jobs_by_day(fs, day_work_time):
    """
    Aggregate the precomputed ``fs`` segments into station-day statistics and a
    station-level summary.
    """

    required_columns = [
        "Assembly Reference",
        "station",
        "Product start time",
        "Product end time",
        "segment_start",
        "segment_end",
        "date",
        "processing_time_sec",
    ]
    missing_columns = [col for col in required_columns if col not in fs.columns]
    if missing_columns:
        raise ValueError("fs is missing required columns: " + ", ".join(missing_columns))

    fs = fs.copy()
    fs["Product start time"] = pd.to_datetime(fs["Product start time"])
    fs["Product end time"] = pd.to_datetime(fs["Product end time"])
    fs["segment_start"] = pd.to_datetime(fs["segment_start"])
    fs["segment_end"] = pd.to_datetime(fs["segment_end"])
    fs["date"] = pd.to_datetime(fs["date"])
    fs["station"] = fs["station"].astype(int)

    if fs.empty:
        station_day_stats = pd.DataFrame(
            columns=[
                "date",
                "station",
                "total_processing_time_sec",
                "jobs_processed",
                "jobs_completed",
                "avg_utilisation",
            ]
        )
        station_summary = pd.DataFrame(
            columns=[
                "station",
                "total_items_processed",
                "min_utilisation",
                "max_utilisation",
                "avg_utilisation",
            ]
        )
        return station_day_stats, station_summary

    station_day_stats = (
        fs.groupby(["date", "station"], as_index=False)
        .agg(total_processing_time_sec=("processing_time_sec", "sum"))
        .sort_values(["date", "station"])
    )

    jobs_processed_by_day = (
        fs.groupby(["date", "station"], as_index=False)
        .agg(jobs_processed=("Assembly Reference", "nunique"))
    )

    jobs_completed_by_day = (
        fs[fs["segment_end"] == fs["Product end time"]]
        .groupby(["date", "station"], as_index=False)
        .agg(jobs_completed=("Assembly Reference", "nunique"))
    )

    station_day_stats = station_day_stats.merge(
        jobs_processed_by_day,
        on=["date", "station"],
        how="left",
    ).merge(
        jobs_completed_by_day,
        on=["date", "station"],
        how="left",
    )

    station_day_stats["jobs_processed"] = (
        station_day_stats["jobs_processed"].fillna(0).astype(int)
    )
    station_day_stats["jobs_completed"] = (
        station_day_stats["jobs_completed"].fillna(0).astype(int)
    )
    station_day_stats["avg_utilisation"] = (
        station_day_stats["total_processing_time_sec"] / day_work_time
    )

    station_summary = (
        station_day_stats.groupby("station", as_index=False)
        .agg(
            total_items_processed=("jobs_processed", "sum"),
            min_utilisation=("avg_utilisation", "min"),
            max_utilisation=("avg_utilisation", "max"),
            avg_utilisation=("avg_utilisation", "mean"),
        )
        .sort_values("station")
    )

    return station_day_stats, station_summary
