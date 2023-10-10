#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=transformer_model \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/home/scratch/brianyan/nuplan_exp/exp/transformer_v2/augmentation_v2/2023.04.18.04.10.45/checkpoints/epoch\=25.ckpt" \
    scenario_builder=nuplan_challenge \
    scenario_filter=nuplan_challenge_scenarios \
    scenario_filter.num_scenarios_per_type=20 \
    number_of_gpus_allocated_per_simulation=0.06 \
    worker.threads_per_node=60