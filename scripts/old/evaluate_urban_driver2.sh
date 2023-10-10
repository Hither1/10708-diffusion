#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_open_loop_model \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/home/scratch/brianyan/nuplan_exp/exp/urban_driver/our_aug/2023.05.06.19.28.20/checkpoints/epoch\=110.ckpt" \
    scenario_builder=nuplan_challenge \
    scenario_filter=nuplan_challenge_scenarios \
    scenario_filter.num_scenarios_per_type=10 \
    number_of_gpus_allocated_per_simulation=0.15 \
    worker.threads_per_node=24