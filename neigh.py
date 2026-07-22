import numpy as np
import pandas as pd


def neighbour_staffing_plan(staffing_plan, start_date, max_workers_per_station=4):
    """
    Generate a neighbouring staffing plan for Simulated Annealing.

    The staffing plan is a matrix with shape (days, stations) where each cell
    indicates the number of workers assigned to a station on a given day.

    This neighbourhood operator performs a *small local change* by either:
        - Increasing the number of workers on a random (day, station), or
        - Decreasing the number of workers on a random (day, station).

    Key characteristics of the move:
    - Workers per station are bounded by [0, max_workers_per_station].
    - Only *weekday rows* are modified since weekends are non-working days
      and changes there would not affect processing or labour cost.
    - Increasing workers can either strengthen an already active station
      or open a new station (0 â†’ 1).
    - Decreasing workers can reduce staffing or close a station (1 â†’ 0).

    Parameters
    ----------
    staffing_plan : ndarray (days x stations)
        Current staffing configuration.
    start_date : datetime-like
        First day of the schedule; used to identify weekend rows.
    max_workers_per_station : int
        Maximum number of workers allowed at a station.

    Returns
    -------
    new_plan : ndarray
        Modified staffing plan (one local change applied).
    move_info : dict
        Information describing the move (useful for debugging SA behaviour).
    """

    # Copy the current plan so we do not modify the original solution
    new_plan = staffing_plan.copy()

    start_date = pd.Timestamp(start_date)

    # Identify which rows correspond to weekdays
    weekday_rows = [
        d for d in range(new_plan.shape[0])
        if (start_date + pd.Timedelta(days=d)).weekday() < 5
    ]

    # If there are no valid working days, return unchanged
    if not weekday_rows:
        return new_plan, None

    # Randomly choose the type of modification
    move_type = np.random.choice(["increase", "decrease"])

    # -----------------------------
    # Increase workers move
    # -----------------------------
    if move_type == "increase":

        # Candidate cells that can still accept workers
        candidates = [
            (d, s)
            for d in weekday_rows
            for s in range(new_plan.shape[1])
            if new_plan[d, s] < max_workers_per_station
        ]

        # If no valid candidates exist, return unchanged
        if not candidates:
            return new_plan, None

        # Randomly select one candidate cell
        d, s = candidates[np.random.randint(len(candidates))]
        old_value = new_plan[d, s]

        # Increase staffing
        new_plan[d, s] += 1

    # -----------------------------
    # Decrease workers move
    # -----------------------------
    else:

        # Candidate cells where workers can be reduced
        candidates = [
            (d, s)
            for d in weekday_rows
            for s in range(new_plan.shape[1])
            if new_plan[d, s] > 0
        ]

        if not candidates:
            return new_plan, None

        d, s = candidates[np.random.randint(len(candidates))]
        old_value = new_plan[d, s]

        # Reduce staffing
        new_plan[d, s] -= 1

    # Store information about the move (useful for debugging / logging)
    move_info = {
        "type": move_type,
        "day": int(d),
        "station": int(s),
        "old_value": int(old_value),
        "new_value": int(new_plan[d, s]),
    }

    return new_plan, move_info
