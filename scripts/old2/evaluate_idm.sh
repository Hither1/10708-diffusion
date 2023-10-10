#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_open_loop_model \
    planner=idm_planner \
    scenario_builder=nuplan_challenge \
    scenario_filter=nuplan_challenge_scenarios \
    scenario_filter.num_scenarios_per_type=10 \
    experiment_name=evaluations \
    worker.threads_per_node=60