import pandas as pd 
import numpy as np 
import pickle
from initial_solution import jigs_initial_sequence
from cost_fnc_jigs import evaluate_schedule_jigs
from adv_neigh import build_move_context, neighbour_staffing_plan_advanced
from starting_temp import estimate_T0_pareto
from MT_SA import SA_pareto

######## Read my input data #####################
#################################################
data=pd.read_csv(r"Berwick_full_deadlines.csv")

data['deadline'] = pd.to_datetime(data['deadline'], dayfirst=True, errors='coerce')
start_date = data['deadline'].min() - pd.Timedelta(days=7) # I suppose for all my products the design is ready one week in advance 
data['DeadlineSeconds'] = (data['deadline'] - start_date).dt.total_seconds().astype(float)



##########################################################################################
#############################  SET OF PARAMETERS #########################################



# SIMULATED ANNEALING PARAMETERS 
T0 = 1.5e9
M=50
N=10*len(data)
alpha=0.988

# Problem parameters

multi_workers_efficiency=0.9
Day_work_time= 8 * 3600
max_jigs=4
jigs_capacity=4
staff_cost=20

family_move_probs = {
    "redistribute": 0.8,
    "open_station": 0.2,
    "close_station": 0,
}

redistribution_move_probs = {
    "rebalance_day_active": 0.38,
    "shift_same_station": 0.30,
    "single_cell_active": 0.22,
    "station_block_active": 0.10,
}

opening_move_probs = {
    "open_station_day": 0.35,
    "open_station_block": 0.65,
}

closing_move_probs = {
    "close_station_day": 0.85,
    "close_station_block": 0.15,
}

######### Generate an initial solution ################################
########################################################################

job_sequence_init, staffing_plan_init, data_sorted_init= jigs_initial_sequence(
    data,
    start_date,
    efficiency_alpha=multi_workers_efficiency,
    T_max=Day_work_time,
    n_stations=max_jigs,
    active_stations=2,
    workers_per_active_station=2,
)


initial_solution = {
    "job_sequence": job_sequence_init,
    "staffing_plan": staffing_plan_init,
}

initial_output = evaluate_schedule_jigs(
    job_sequence=initial_solution["job_sequence"],
    staffing_plan=initial_solution["staffing_plan"],
    data=data_sorted_init,
    start_date=start_date,
    worker_hourly_cost=staff_cost,
    T_max=Day_work_time,
    efficiency_alpha=multi_workers_efficiency,
    max_workers_per_station=jigs_capacity,
    n_stations=max_jigs,
)

initial_solution["station_assignment"] = initial_output["station_assignment"].copy()
initial_move_context = build_move_context(
    staffing_plan=initial_solution["staffing_plan"],
    job_sequence=initial_solution["job_sequence"],
    station_assignment=initial_output["station_assignment"],
    tardiness_vector=initial_output["tardiness_vector"],
    data=data_sorted_init,
    start_date=start_date,
    T_max=Day_work_time,
    efficiency_alpha=multi_workers_efficiency,
)
initial_output["station_day_utilization"] = initial_move_context["station_day_utilization"].copy()

with open("initial_solution.pkl", "wb") as f:
    pickle.dump(initial_solution, f)

with open("initial_output.pkl", "wb") as f:
    pickle.dump(initial_output, f)



######### Set the temperature  optimal ################################
########################################################################

def evaluate_solution(solution):
    return evaluate_schedule_jigs(
        job_sequence=solution["job_sequence"],
        staffing_plan=solution["staffing_plan"],
        data=data_sorted_init,
        start_date=start_date,
        worker_hourly_cost=staff_cost,
        T_max=Day_work_time,
        efficiency_alpha=multi_workers_efficiency,
        max_workers_per_station=jigs_capacity,
        n_stations=max_jigs,
        station_assignment=solution.get("station_assignment"),
    )


def neighbour_solution(solution):
    new_sequence = solution["job_sequence"].copy()

    new_staffing, move_info = neighbour_staffing_plan_advanced(
        staffing_plan=solution["staffing_plan"],
        start_date=start_date,
        max_workers_per_station=jigs_capacity,
        family_probs=family_move_probs,
        redistribution_probs=redistribution_move_probs,
        opening_probs=opening_move_probs,
        closing_probs=closing_move_probs,
        move_context=initial_move_context,
    )

    new_solution = {
        "job_sequence": new_sequence,
        "staffing_plan": new_staffing,
    }

    if move_info is not None and not move_info.get("recompute_assignment", False):
        new_solution["station_assignment"] = solution["station_assignment"].copy()

    return new_solution, move_info


T0_est, T0_details = estimate_T0_pareto(
    initial_solution=initial_solution,
    evaluate_solution=evaluate_solution,
    neighbour_function=neighbour_solution,
    n_samples=200,
    target_acceptance=0.8,
    scale_mode="median_abs_change",
)

print("Estimated T0:", T0_est)
print("Initial objective vector [labour_cost, total_tardiness, severity]:", T0_details["current_obj"])
print("Objective scales:", T0_details["scales"])
print("Median positive deterioration:", T0_details["median_positive_delta"])
print("Saved initial solution to initial_solution.pkl")
print("Saved initial output to initial_output.pkl")



######### SIMULATED ANNEALING LOOP #####################################
########################################################################

final_solution, final_result, pareto_archive, log_file = SA_pareto(
    data=data_sorted_init,
    initial_solution=initial_solution,
    start_date=start_date,
    worker_hourly_cost=staff_cost,
    T0=T0_est,
    alpha=alpha,
    M=M,
    N=N,
    T_max=Day_work_time,
    efficiency_alpha=multi_workers_efficiency,
    max_workers_per_station=jigs_capacity,
    n_stations=max_jigs,
    objective_scales=T0_details["scales"],
    family_probs=family_move_probs,
    redistribution_probs=redistribution_move_probs,
    opening_probs=opening_move_probs,
    closing_probs=closing_move_probs,
    num_cores=None,
    chunk_multiplier=30,
    log_dir="."
)

archive_file = "pareto_archive.pkl"
with open(archive_file, "wb") as f:
    pickle.dump(pareto_archive, f)

summary = pd.DataFrame([
    {
        "cost": entry["objectives"][0],
        "tardiness": entry["objectives"][1],
        "severity": entry["objectives"][2],
    }
    for entry in pareto_archive
])

summary.to_csv("pareto_frontier_summary.csv", index=False)

print("Final current objectives:")
print(final_result["labour_cost"], final_result["total_tardiness"], final_result["severity"])
print("Pareto archive size:", len(pareto_archive))
print("Log file:", log_file)
