import os
import ast
import pandas
from overcooked_ai_pcg import LSI_HUMAN_STUDY_RESULT_DIR

from overcooked_ai_pcg.LSI.human_study import (DETAILED_STUDY_TYPES,
                                               load_human_log_data)

merged_human_logs = pandas.DataFrame()

bc_column_names = [
    "total_sparse_reward",
    "checkpoints",
    "workloads",
    "concurr_active",
    "stuck_time",
]

direct_mean_column_names = [
    "total_sparse_reward",
    "concurr_active",
    "stuck_time",
]

for bc_column_name in bc_column_names:
    first_log_index = None
    last_log_index = None
    num_log_dir = 0
    for i, log_index in enumerate(
            sorted(os.listdir(LSI_HUMAN_STUDY_RESULT_DIR))):
        if not os.path.isdir(
                os.path.join(LSI_HUMAN_STUDY_RESULT_DIR, log_index)):
            continue
        _, human_log_data = load_human_log_data(log_index)
        human_log_data = human_log_data.sort_values(by=["lvl_type"])
        # add lvl types
        if not "lvl_type" in merged_human_logs:
            merged_human_logs["lvl_type"] = human_log_data["lvl_type"]

        # add bc
        merged_human_logs[
            f"user-{log_index}-{bc_column_name}"] = human_log_data[
                bc_column_name]

        if num_log_dir == 0:
            first_log_index = log_index

        num_log_dir += 1
        last_log_index = log_index

    # add mean of the added bc as the new colunmn
    num_user = len(os.listdir(LSI_HUMAN_STUDY_RESULT_DIR))

    sub_cols = merged_human_logs.loc[:,
                                     f"user-{first_log_index}-{bc_column_name}":
                                     f"user-{last_log_index}-{bc_column_name}"]

    if bc_column_name in direct_mean_column_names:

        merged_human_logs[f'{bc_column_name}_mean'] = sub_cols.mean(axis=1)

    elif bc_column_name == "workloads":
        # for every level
        all_lvl_avg_workloads = []
        for index, row in sub_cols.iterrows():
            avg_workloads = [{
                'num_ingre_held': 0,
                'num_plate_held': 0,
                'num_served': 0
            }, {
                'num_ingre_held': 0,
                'num_plate_held': 0,
                'num_served': 0
            }]
            # take the average of the workloads of all users
            for user_col in row:
                workload = ast.literal_eval(user_col)
                for j, agent_workload in enumerate(avg_workloads):
                    for key in agent_workload:
                        avg_workloads[j][key] += workload[j][key]

            # take the average
            for j, agent_workload in enumerate(avg_workloads):
                for key in agent_workload:
                    avg_workloads[j][key] /= row.shape[0]

            all_lvl_avg_workloads.append(avg_workloads)

        merged_human_logs[f'{bc_column_name}_mean'] = all_lvl_avg_workloads


merged_human_logs.to_csv(os.path.join(LSI_HUMAN_STUDY_RESULT_DIR,
                                      "merged_human_log.csv"),
                         index=False)
