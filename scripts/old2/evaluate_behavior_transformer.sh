#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=behavior_transformer \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/behavior_transformer/checkpoints/epoch\=399.ckpt" \
    scenario_builder=nuplan_challenge \
    scenario_filter=nuplan_challenge_scenarios \
    scenario_filter.num_scenarios_per_type=10 \
    number_of_gpus_allocated_per_simulation=0.1 \
    experiment_name=evaluations \
    worker.threads_per_node=60