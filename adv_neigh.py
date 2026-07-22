import numpy as np
import pandas as pd


def _weekday_rows(n_days, start_date):
    start_date = pd.Timestamp(start_date)
    return [
        d for d in range(n_days)
        if (start_date + pd.Timedelta(days=d)).weekday() < 5
    ]


def _focused_rows(plan, weekday_rows, look_ahead_days=3):
    active_rows = [d for d in weekday_rows if np.any(plan[d, :] > 0)]
    if not active_rows:
        return weekday_rows

    max_row = min(plan.shape[0] - 1, max(active_rows) + look_ahead_days)
    return [d for d in weekday_rows if d <= max_row]


def _editable_weekday_rows(weekday_rows, move_context, buffer_days=3):
    if not weekday_rows:
        return []
    if move_context is None:
        return list(weekday_rows)

    last_used_day = move_context.get("last_used_day")
    if last_used_day is None:
        return list(weekday_rows)

    cutoff_day = int(last_used_day) + int(buffer_days)
    editable = [d for d in weekday_rows if d <= cutoff_day]
    return editable if editable else [weekday_rows[0]]


def _normalize_probs(prob_dict, expected_keys, dict_name):
    probs = np.array([float(prob_dict[key]) for key in expected_keys], dtype=float)
    if np.any(probs < 0) or probs.sum() <= 0:
        raise ValueError(f"{dict_name} must contain non-negative values with positive total.")
    return probs / probs.sum()


def _active_station_mask(plan):
    return np.any(plan > 0, axis=0)


def _choose_move_type(rng, family_probs, redistribution_probs, opening_probs, closing_probs):
    family_names = ["redistribute", "open_station", "close_station"]
    family_p = _normalize_probs(family_probs, family_names, "family_probs")
    family = str(rng.choice(family_names, p=family_p))

    if family == "redistribute":
        move_names = [
            "rebalance_day_active",
            "shift_same_station",
            "single_cell_active",
            "station_block_active",
        ]
        move_p = _normalize_probs(redistribution_probs, move_names, "redistribution_probs")
        return family, str(rng.choice(move_names, p=move_p))

    if family == "open_station":
        move_names = ["open_station_day", "open_station_block"]
        move_p = _normalize_probs(opening_probs, move_names, "opening_probs")
        return family, str(rng.choice(move_names, p=move_p))

    move_names = ["close_station_day", "close_station_block"]
    move_p = _normalize_probs(closing_probs, move_names, "closing_probs")
    return family, str(rng.choice(move_names, p=move_p))


def build_move_context(
    staffing_plan,
    job_sequence,
    station_assignment,
    tardiness_vector,
    data,
    start_date,
    T_max,
    efficiency_alpha,
):
    """
    Build a compact move-guidance context from the current evaluated solution.

    The context is computed once for the current solution and then reused by
    neighbour generation so we can bias moves toward actual pressure days and
    slack station-days without re-running a full reconstruction for every
    sampled neighbour.
    """
    staffing_plan = np.asarray(staffing_plan, dtype=int)
    job_sequence = np.asarray(job_sequence, dtype=int)
    station_assignment = np.asarray(station_assignment, dtype=int)
    tardiness_vector = np.asarray(tardiness_vector, dtype=float)
    start_date = pd.Timestamp(start_date)

    if staffing_plan.ndim != 2:
        raise ValueError("staffing_plan must be a 2D array.")

    n_days, n_stations = staffing_plan.shape
    n_jobs = len(job_sequence)
    if len(station_assignment) != n_jobs or len(tardiness_vector) != n_jobs:
        raise ValueError("station_assignment and tardiness_vector must match the job sequence length.")

    seconds_in_day = 24 * 3600
    weekday_rows = _weekday_rows(n_days, start_date)
    station_available_time = np.zeros(n_stations, dtype=float)
    station_day_work = np.zeros((n_days, n_stations), dtype=float)
    tardy_day_scores = np.zeros(n_days, dtype=float)
    slack_day_scores = np.zeros(n_days, dtype=float)

    dimple_counts = data["DIMPLE_count"].to_numpy(dtype=float)
    if "DeadlineSeconds" in data.columns:
        deadline_seconds = data["DeadlineSeconds"].to_numpy(dtype=float)
    elif "deadline" in data.columns:
        deadline_seconds = (
            pd.to_datetime(data["deadline"], dayfirst=True, errors="coerce") - start_date
        ).dt.total_seconds().to_numpy(dtype=float)
    else:
        deadline_seconds = np.full(n_jobs, np.inf, dtype=float)

    def is_weekend(day_idx):
        return (start_date + pd.Timedelta(days=int(day_idx))).weekday() >= 5

    def next_weekday(day_idx):
        while is_weekend(day_idx):
            day_idx += 1
        return day_idx

    for job_idx in job_sequence:
        station_idx = int(station_assignment[job_idx])
        base_work = float(dimple_counts[job_idx]) * 60.0
        if base_work <= 0:
            continue

        current_time = float(station_available_time[station_idx])
        remaining_work = float(base_work)
        job_tardiness = max(0.0, float(tardiness_vector[job_idx]))
        job_segments = []

        while True:
            day_idx = int(current_time // seconds_in_day)
            time_in_day = current_time % seconds_in_day

            if day_idx >= n_days:
                break

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
            work_done_today = processing_time_today * speed_today

            station_day_work[day_idx, station_idx] += work_done_today
            job_segments.append((day_idx, work_done_today))
            if job_tardiness > 0:
                tardy_day_scores[day_idx] += job_tardiness * (work_done_today / base_work)

            remaining_work -= work_done_today
            if remaining_work <= 1e-9:
                completion_time = day_idx * seconds_in_day + time_in_day + processing_time_today
                station_available_time[station_idx] = completion_time
                deadline_value = float(deadline_seconds[job_idx])
                job_slack = 0.0 if not np.isfinite(deadline_value) else max(0.0, deadline_value - completion_time)
                if job_slack > 0:
                    for seg_day, seg_work in job_segments:
                        slack_day_scores[seg_day] += job_slack * (seg_work / base_work)
                break

            current_time = (day_idx + 1) * seconds_in_day

    station_day_capacity = np.zeros((n_days, n_stations), dtype=float)
    for d in weekday_rows:
        positive_mask = staffing_plan[d, :] > 0
        if np.any(positive_mask):
            station_day_capacity[d, positive_mask] = (
                T_max * np.power(staffing_plan[d, positive_mask].astype(float), efficiency_alpha)
            )

    station_day_utilization = np.zeros((n_days, n_stations), dtype=float)
    positive_capacity = station_day_capacity > 0
    station_day_utilization[positive_capacity] = (
        station_day_work[positive_capacity] / station_day_capacity[positive_capacity]
    )
    last_completion = float(np.max(station_available_time)) if n_stations > 0 else 0.0
    last_used_day = int(np.floor(last_completion / seconds_in_day)) if last_completion > 0 else 0

    return {
        "tardy_day_scores": tardy_day_scores,
        "slack_day_scores": slack_day_scores,
        "station_day_utilization": station_day_utilization,
        "last_used_day": last_used_day,
    }


def _day_scores(move_context, n_days):
    if move_context is None:
        return np.zeros(n_days, dtype=float)
    values = np.asarray(move_context.get("tardy_day_scores"), dtype=float)
    if values.shape != (n_days,):
        return np.zeros(n_days, dtype=float)
    return values


def _station_utilization(move_context, shape):
    if move_context is None:
        return np.zeros(shape, dtype=float)
    values = np.asarray(move_context.get("station_day_utilization"), dtype=float)
    if values.shape != shape:
        return np.zeros(shape, dtype=float)
    return values


def _slack_scores(move_context, n_days):
    if move_context is None:
        return np.zeros(n_days, dtype=float)
    values = np.asarray(move_context.get("slack_day_scores"), dtype=float)
    if values.shape != (n_days,):
        return np.zeros(n_days, dtype=float)
    return values


def _normalize_signal(values):
    values = np.asarray(values, dtype=float)
    max_value = float(np.max(values)) if values.size > 0 else 0.0
    if max_value <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return values / max_value


def _expanded_day_scores(day_scores, ordered_rows, backward_window=5, backward_decay=0.80, forward_spill=0.15):
    expanded = np.asarray(day_scores, dtype=float).copy()
    for idx, d in enumerate(ordered_rows):
        base = float(day_scores[d])
        if base <= 0:
            continue
        for step in range(1, backward_window + 1):
            source_idx = idx - step
            if source_idx < 0:
                break
            expanded[ordered_rows[source_idx]] += (backward_decay ** step) * base
        if idx + 1 < len(ordered_rows):
            expanded[ordered_rows[idx + 1]] += forward_spill * base
    return expanded


def _weighted_row_choice(rng, candidates, row_weights):
    if not candidates:
        return None

    weights = np.array([max(0.0, float(row_weights[d])) for d in candidates], dtype=float)
    if weights.sum() <= 0:
        return int(rng.choice(candidates))
    return int(rng.choice(candidates, p=weights / weights.sum()))


def _collect_blocks(ordered_rows, eligible_rows, min_len=2, max_len=3):
    eligible_set = set(int(d) for d in eligible_rows)
    blocks = []
    run = []

    def flush(current_run):
        if len(current_run) < min_len:
            return
        upper = min(max_len, len(current_run))
        for block_len in range(min_len, upper + 1):
            for start_idx in range(len(current_run) - block_len + 1):
                blocks.append(current_run[start_idx:start_idx + block_len])

    for d in ordered_rows:
        if d in eligible_set:
            run.append(int(d))
        else:
            flush(run)
            run = []
    flush(run)

    return blocks


def _choose_high_worker_count(rng, max_workers_per_station, day_score, all_day_scores):
    allowed = np.arange(1, max_workers_per_station + 1, dtype=int)
    if len(allowed) == 1:
        return int(allowed[0])

    positive_scores = np.asarray(all_day_scores, dtype=float)
    positive_scores = positive_scores[positive_scores > 0]
    reference = float(np.median(positive_scores)) if positive_scores.size > 0 else 0.0
    power = 2.3 if day_score >= reference and reference > 0 else 1.8
    weights = np.power(allowed.astype(float), power)
    weights[0] *= 0.12
    return int(rng.choice(allowed, p=weights / weights.sum()))


def _fallback_bottleneck_days(candidate_plan, focused_rows, max_workers_per_station):
    active_mask = _active_station_mask(candidate_plan)
    active_stations = [s for s, is_active in enumerate(active_mask) if is_active]
    if not active_stations:
        return []

    bottlenecks = []
    for d in focused_rows:
        staffed_active = [candidate_plan[d, s] for s in active_stations]
        if staffed_active and max(staffed_active) >= max_workers_per_station:
            bottlenecks.append(d)
    return bottlenecks


def _pick_station_for_opening(candidate_plan, days, rng):
    n_stations = candidate_plan.shape[1]
    candidates = [
        s for s in range(n_stations)
        if all(candidate_plan[d, s] == 0 for d in days)
    ]
    if not candidates:
        return None

    day_counts = np.count_nonzero(candidate_plan > 0, axis=0)
    best_count = int(np.min(day_counts[candidates]))
    best = [int(s) for s in candidates if day_counts[s] == best_count]
    return int(rng.choice(best))


def _apply_redistribution_move(
    candidate_plan,
    move_type,
    focused_rows,
    weekday_rows,
    max_workers_per_station,
    move_context,
    rng,
):
    n_days, n_stations = candidate_plan.shape
    active_mask = _active_station_mask(candidate_plan)
    active_stations = [s for s in range(n_stations) if active_mask[s]]
    day_scores = _day_scores(move_context, n_days)
    slack_scores = _slack_scores(move_context, n_days)
    pressure_scores = _expanded_day_scores(day_scores, weekday_rows)
    pressure_norm = _normalize_signal(pressure_scores)
    slack_norm = _normalize_signal(slack_scores)
    receiver_day_scores = pressure_norm + 0.20 * (1.0 - slack_norm)
    safe_source_day_scores = np.where(pressure_scores <= 1e-9, slack_norm, 0.0)
    station_util = _station_utilization(move_context, candidate_plan.shape)

    def info(move_type_name, **kwargs):
        out = {"family": "redistribute", "type": move_type_name, "recompute_assignment": False}
        out.update(kwargs)
        return out

    if not active_stations:
        return None, None

    if move_type == "rebalance_day_active":
        day_candidates = [
            d for d in focused_rows
            if sum(candidate_plan[d, s] > 0 for s in active_stations) >= 2
            and any(candidate_plan[d, s] > 1 for s in active_stations)
            and any(0 < candidate_plan[d, s] < max_workers_per_station for s in active_stations)
        ]
        if not day_candidates:
            return None, None

        d = _weighted_row_choice(rng, day_candidates, receiver_day_scores)
        from_candidates = [s for s in active_stations if candidate_plan[d, s] > 1]
        to_candidates = [
            s for s in active_stations
            if 0 < candidate_plan[d, s] < max_workers_per_station
        ]
        if not from_candidates or not to_candidates:
            return None, None

        valid_from_candidates = [s for s in from_candidates if any(t != s for t in to_candidates)]
        if not valid_from_candidates:
            return None, None

        s_from = min(
            valid_from_candidates,
            key=lambda s: (candidate_plan[d, s], station_util[d, s]),
        )
        to_candidates = [s for s in to_candidates if s != s_from]
        if not to_candidates:
            return None, None

        s_to = max(
            to_candidates,
            key=lambda s: (candidate_plan[d, s], station_util[d, s]),
        )
        candidate_plan[d, s_from] -= 1
        candidate_plan[d, s_to] += 1
        return candidate_plan, info("rebalance_day_active", day=d, station_from=s_from, station_to=s_to)

    if move_type == "shift_same_station":
        feasible = []
        for s in active_stations:
            source_days = [
                d for d in weekday_rows
                if candidate_plan[d, s] > 1 and safe_source_day_scores[d] > 0
            ]
            target_days = [
                d for d in focused_rows
                if 0 < candidate_plan[d, s] < max_workers_per_station and pressure_scores[d] > 0
            ]
            if source_days and target_days:
                feasible.append((s, source_days, target_days))

        if not feasible:
            return None, None

        s, source_days, target_days = max(
            feasible,
            key=lambda item: max(receiver_day_scores[d] for d in item[2]),
        )
        d_to = _weighted_row_choice(rng, target_days, receiver_day_scores)
        nearby_sources = [d for d in source_days if d != d_to and abs(d - d_to) <= 3]
        source_pool = nearby_sources if nearby_sources else [d for d in source_days if d != d_to]
        if not source_pool:
            return None, None

        source_weights = np.array(
            [safe_source_day_scores[d] / (1.0 + 0.15 * abs(d - d_to)) for d in source_pool],
            dtype=float,
        )
        if source_weights.sum() <= 0:
            d_from = int(rng.choice(source_pool))
        else:
            d_from = int(rng.choice(source_pool, p=source_weights / source_weights.sum()))
        candidate_plan[d_from, s] -= 1
        candidate_plan[d_to, s] += 1
        return candidate_plan, info("shift_same_station", station=s, day_from=d_from, day_to=d_to)

    if move_type == "single_cell_active":
        increase_candidates = [
            (d, s)
            for d in focused_rows
            for s in active_stations
            if pressure_scores[d] > 0 and 0 < candidate_plan[d, s] < max_workers_per_station
        ]
        decrease_candidates = [
            (d, s)
            for d in weekday_rows
            for s in active_stations
            if safe_source_day_scores[d] > 0 and candidate_plan[d, s] > 1
        ]
        if not increase_candidates and not decrease_candidates:
            return None, None

        choose_increase = bool(increase_candidates) and (
            not decrease_candidates or rng.random() < 0.65
        )
        direction = "increase" if choose_increase else "decrease"
        candidates = increase_candidates if choose_increase else decrease_candidates

        if direction == "increase":
            weights = np.array(
                [receiver_day_scores[d] * (0.5 + station_util[d, s]) for d, s in candidates],
                dtype=float,
            )
        else:
            weights = np.array(
                [safe_source_day_scores[d] * (1.0 + 0.10 * (1.0 - station_util[d, s])) for d, s in candidates],
                dtype=float,
            )

        weights = weights / weights.sum()
        d, s = candidates[int(rng.choice(len(candidates), p=weights))]
        old_value = int(candidate_plan[d, s])
        candidate_plan[d, s] += 1 if direction == "increase" else -1
        return candidate_plan, info(
            "single_cell_active",
            direction=direction,
            day=int(d),
            station=int(s),
            old_value=old_value,
            new_value=int(candidate_plan[d, s]),
        )

    if move_type == "station_block_active":
        increase_blocks = _collect_blocks(
            focused_rows,
            [d for d in focused_rows if pressure_scores[d] > 0],
            min_len=2,
            max_len=3,
        )
        decrease_blocks = _collect_blocks(
            weekday_rows,
            [d for d in weekday_rows if safe_source_day_scores[d] > 0],
            min_len=2,
            max_len=3,
        )
        if not increase_blocks and not decrease_blocks:
            return None, None

        choose_increase = bool(increase_blocks) and (
            not decrease_blocks or rng.random() < 0.70
        )
        direction = "increase_block" if choose_increase else "decrease_block"
        blocks = increase_blocks if choose_increase else decrease_blocks

        block_candidates = []
        for block in blocks:
            for s in active_stations:
                if direction == "increase_block":
                    if all(0 < candidate_plan[d, s] < max_workers_per_station for d in block):
                        score = sum(receiver_day_scores[d] for d in block) + 0.10 * sum(station_util[d, s] for d in block)
                        block_candidates.append((score, s, block))
                else:
                    if all(candidate_plan[d, s] > 1 for d in block):
                        score = sum(safe_source_day_scores[d] for d in block) + 0.10 * sum(1.0 - station_util[d, s] for d in block)
                        block_candidates.append((score, s, block))
        if not block_candidates:
            return None, None

        if direction == "increase_block":
            _, s, block = max(block_candidates, key=lambda item: item[0])
        else:
            _, s, block = max(block_candidates, key=lambda item: item[0])

        for d in block:
            candidate_plan[d, s] += 1 if direction == "increase_block" else -1

        return candidate_plan, info("station_block_active", direction=direction, station=s, days=[int(d) for d in block])

    return None, None


def _apply_opening_move(candidate_plan, move_type, focused_rows, max_workers_per_station, move_context, rng):
    n_days, _ = candidate_plan.shape
    day_scores = _day_scores(move_context, n_days)
    pressure_scores = _expanded_day_scores(day_scores, focused_rows)

    def info(move_type_name, **kwargs):
        out = {"family": "open_station", "type": move_type_name, "recompute_assignment": True}
        out.update(kwargs)
        return out

    if move_type == "open_station_day":
        day_candidates = [
            d for d in focused_rows
            if pressure_scores[d] > 0 and np.any(candidate_plan[d, :] == 0)
        ]
        if not day_candidates:
            fallback_days = _fallback_bottleneck_days(candidate_plan, focused_rows, max_workers_per_station)
            day_candidates = [d for d in fallback_days if np.any(candidate_plan[d, :] == 0)]
        if not day_candidates:
            return None, None

        d = _weighted_row_choice(rng, day_candidates, pressure_scores)
        opened_station = _pick_station_for_opening(candidate_plan, [d], rng)
        if opened_station is None:
            return None, None

        workers = _choose_high_worker_count(rng, max_workers_per_station, pressure_scores[d], pressure_scores)
        candidate_plan[d, opened_station] = workers
        return candidate_plan, info("open_station_day", station=opened_station, day=d, workers=workers)

    if move_type == "open_station_block":
        blocks = _collect_blocks(
            focused_rows,
            [d for d in focused_rows if pressure_scores[d] > 0],
            min_len=2,
            max_len=3,
        )
        if not blocks:
            return None, None

        weighted_blocks = []
        for block in blocks:
            opened_station = _pick_station_for_opening(candidate_plan, block, rng)
            if opened_station is None:
                continue
            weighted_blocks.append((sum(pressure_scores[d] for d in block), opened_station, block))
        if not weighted_blocks:
            return None, None

        _, opened_station, block = max(weighted_blocks, key=lambda item: item[0])
        block_score = max(pressure_scores[d] for d in block)
        workers = _choose_high_worker_count(rng, max_workers_per_station, block_score, pressure_scores)
        for d in block:
            candidate_plan[d, opened_station] = workers
        return candidate_plan, info("open_station_block", station=opened_station, days=[int(d) for d in block], workers=workers)

    return None, None


def _apply_closing_move(candidate_plan, move_type, focused_rows, move_context, rng):
    n_days, n_stations = candidate_plan.shape
    day_scores = _day_scores(move_context, n_days)
    slack_scores = _slack_scores(move_context, n_days)
    pressure_scores = _expanded_day_scores(day_scores, focused_rows)
    safe_closing_scores = np.where(pressure_scores <= 1e-9, _normalize_signal(slack_scores), 0.0)
    station_util = _station_utilization(move_context, candidate_plan.shape)
    active_mask = _active_station_mask(candidate_plan)
    active_stations = [s for s in range(n_stations) if active_mask[s]]

    def info(move_type_name, **kwargs):
        out = {"family": "close_station", "type": move_type_name, "recompute_assignment": True}
        out.update(kwargs)
        return out

    if not active_stations:
        return None, None

    if move_type == "close_station_day":
        candidates = []
        for d in focused_rows:
            if safe_closing_scores[d] <= 0 or np.count_nonzero(candidate_plan[d, :] > 0) <= 1:
                continue
            for s in active_stations:
                if candidate_plan[d, s] > 0:
                    candidates.append((d, s))
        if not candidates:
            return None, None

        d, s = max(
            candidates,
            key=lambda item: (
                safe_closing_scores[item[0]],
                1.0 - station_util[item[0], item[1]],
                candidate_plan[item[0], item[1]],
            ),
        )
        old_value = int(candidate_plan[d, s])
        candidate_plan[d, s] = 0
        return candidate_plan, info("close_station_day", station=int(s), day=int(d), old_value=old_value, new_value=0)

    if move_type == "close_station_block":
        blocks = _collect_blocks(
            focused_rows,
            [d for d in focused_rows if safe_closing_scores[d] > 0],
            min_len=2,
            max_len=3,
        )
        if not blocks:
            return None, None

        block_candidates = []
        for block in blocks:
            for s in active_stations:
                if all(
                    candidate_plan[d, s] > 0
                    and np.count_nonzero(candidate_plan[d, :] > 0) > 1
                    for d in block
                ):
                    score = sum(safe_closing_scores[d] for d in block) + 0.10 * sum(
                        1.0 - station_util[d, s] for d in block
                    )
                    block_candidates.append((score, s, block))
        if not block_candidates:
            return None, None

        _, s, block = max(block_candidates, key=lambda item: item[0])
        old_values = [int(candidate_plan[d, s]) for d in block]
        for d in block:
            candidate_plan[d, s] = 0
        return candidate_plan, info("close_station_block", station=int(s), days=[int(d) for d in block], old_values=old_values, new_value=0)

    return None, None


def neighbour_staffing_plan_advanced(
    staffing_plan,
    start_date,
    max_workers_per_station=4,
    look_ahead_days=3,
    family_probs=None,
    redistribution_probs=None,
    opening_probs=None,
    closing_probs=None,
    move_context=None,
):
    """
    Advanced staffing neighbourhood with three move families:
    - redistribution among already active station-days
    - opening station-days or short station blocks on tardy windows
    - closing station-days or short station blocks on slack windows

    Returns
    -------
    new_plan : np.ndarray
        Modified staffing plan.
    move_info : dict or None
        Metadata describing the applied move.
    """
    if family_probs is None:
        family_probs = {
            "redistribute": 0.70,
            "open_station": 0.15,
            "close_station": 0.15,
        }
    if redistribution_probs is None:
        redistribution_probs = {
            "rebalance_day_active": 0.38,
            "shift_same_station": 0.30,
            "single_cell_active": 0.22,
            "station_block_active": 0.10,
        }
    if opening_probs is None:
        opening_probs = {
            "open_station_day": 0.35,
            "open_station_block": 0.65,
        }
    if closing_probs is None:
        closing_probs = {
            "close_station_day": 0.85,
            "close_station_block": 0.15,
        }

    new_plan = np.asarray(staffing_plan, dtype=int).copy()
    n_days, _ = new_plan.shape

    if np.any(new_plan < 0) or np.any(new_plan > max_workers_per_station):
        raise ValueError(f"Each staffing entry must be between 0 and {max_workers_per_station}.")

    weekday_rows = _weekday_rows(n_days, start_date)
    if not weekday_rows:
        return new_plan, None

    editable_weekday_rows = _editable_weekday_rows(weekday_rows, move_context, buffer_days=look_ahead_days)
    if not editable_weekday_rows:
        return new_plan, None

    focused_rows = _focused_rows(new_plan, editable_weekday_rows, look_ahead_days=look_ahead_days)
    rng = np.random.default_rng()

    for _ in range(12):
        candidate_plan = new_plan.copy()
        family, move_type = _choose_move_type(
            rng,
            family_probs,
            redistribution_probs,
            opening_probs,
            closing_probs,
        )

        if family == "redistribute":
            candidate_plan, move_info = _apply_redistribution_move(
                candidate_plan=candidate_plan,
                move_type=move_type,
                focused_rows=focused_rows,
                weekday_rows=editable_weekday_rows,
                max_workers_per_station=max_workers_per_station,
                move_context=move_context,
                rng=rng,
            )
        elif family == "open_station":
            candidate_plan, move_info = _apply_opening_move(
                candidate_plan=candidate_plan,
                move_type=move_type,
                focused_rows=focused_rows,
                max_workers_per_station=max_workers_per_station,
                move_context=move_context,
                rng=rng,
            )
        else:
            candidate_plan, move_info = _apply_closing_move(
                candidate_plan=candidate_plan,
                move_type=move_type,
                focused_rows=focused_rows,
                move_context=move_context,
                rng=rng,
            )

        if candidate_plan is None:
            continue

        return candidate_plan, move_info

    return new_plan, None
