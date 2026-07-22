# Type A: 
# Randomly increase or decrease the total number of workers for the day
# Redistribute the new total across stations (add a worker to a station or open a new station if needed)

# Type B: 
# Keep total daily workers constant
# Move 1 worker from one station to another, respecting 1–4 workers per station




import pandas as pd 
import numpy as np 
import random

def neighbour_search_day(day_stations, daily_workers, min_workers=1, max_workers=16):
    """
    Generate a neighbour for a single day under the following constraints:
    - Max 4 workers per station
    - Each station must have at least 1 worker
    - Max total daily workers = 16 (4 stations × 4)
    - Can increase/decrease total daily workers or redistribute them

    Parameters
    ----------
    day_stations : dict
        Current allocation {station_id: workers}
    daily_workers : int
        Current total workers for the day
    min_workers : int
        Minimum daily workers
    max_workers : int
        Maximum daily workers

    Returns
    -------
    new_day_stations : dict
        Updated station worker allocation
    new_daily_workers : int
        Updated total workers for the day
    """
    new_day_stations = day_stations.copy()
    new_daily_workers = daily_workers

    move_type = random.random()

    # -------------------------
    # Type A: Increase/Decrease total daily workers
    # -------------------------
    if move_type < 0.5:
        delta = random.choice([-1, 1])
        tentative_workers = daily_workers + delta

        if min_workers <= tentative_workers <= max_workers:
            new_daily_workers = tentative_workers
            diff = new_daily_workers - sum(new_day_stations.values())
            station_ids = list(new_day_stations.keys())

            # Adjust workers across stations
            while diff != 0:
                if diff > 0:
                    # add worker to existing station or open a new one
                    s = random.randint(0, 3)
                    if s in new_day_stations:
                        if new_day_stations[s] < 4:
                            new_day_stations[s] += 1
                            diff -= 1
                    else:
                        new_day_stations[s] = 1
                        diff -= 1
                else:
                    # remove worker from a station only if > 1
                    s = random.choice(station_ids)
                    if new_day_stations[s] > 1:
                        new_day_stations[s] -= 1
                        diff += 1
                    # do nothing if station has 1 worker (cannot go below 1)

    # -------------------------
    # Type B: Redistribute workers across stations (total fixed)
    # -------------------------
    else:
        station_ids = list(new_day_stations.keys())
        if len(station_ids) >= 2:
            donor, receiver = random.sample(station_ids, 2)

            # move 1 worker if it doesn't violate 1–4 limit
            if new_day_stations[donor] > 1 and new_day_stations[receiver] < 4:
                new_day_stations[donor] -= 1
                new_day_stations[receiver] += 1

            # Optional: open a new station while removing from an existing (still keep ≥1)
            if random.random() < 0.2:
                for s in range(4):
                    if s not in new_day_stations:
                        donor2 = random.choice(station_ids)
                        if new_day_stations[donor2] > 1:
                            new_day_stations[donor2] -= 1
                            new_day_stations[s] = 1
                        break

    return new_day_stations, new_daily_workers





import random

def neighbour_search_schedule(schedule_by_day, daily_workers_by_day,
                              min_workers=1, max_workers=16):
    """
    Generate a neighbour schedule for all production days.

    Parameters
    ----------
    schedule_by_day : dict
        Keys = day (e.g., 0,1,2,...)
        Values = dict {station_id: workers} representing each day's station allocation
    daily_workers_by_day : dict
        Keys = day
        Values = total number of workers used on that day
    min_workers : int
        Minimum daily workers
    max_workers : int
        Maximum daily workers

    Returns
    -------
    new_schedule_by_day : dict
        Updated schedule with neighbour moves applied
    new_daily_workers_by_day : dict
        Updated daily workers counts
    """
    new_schedule_by_day = {}
    new_daily_workers_by_day = {}

    for day, day_stations in schedule_by_day.items():
        daily_workers = daily_workers_by_day[day]
        
        # Apply neighbour move to this day
        new_day_stations, new_daily_workers = neighbour_search_day(
            day_stations, daily_workers, min_workers, max_workers
        )
        
        new_schedule_by_day[day] = new_day_stations
        new_daily_workers_by_day[day] = new_daily_workers

    return new_schedule_by_day, new_daily_workers_by_day