#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_diffusion_model \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion/2023.05.24.15.27.00/checkpoints/epoch\=91.ckpt" \
    scenario_builder=nuplan_challenge \
    scenario_filter=nuplan_challenge_scenarios \
    scenario_filter.num_scenarios_per_type=10 \
    number_of_gpus_allocated_per_simulation=0.24 \
    experiment_name=evaluations \
    worker.threads_per_node=60