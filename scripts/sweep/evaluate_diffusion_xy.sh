#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_diffusion_model \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion/2023.07.04.14.40.09/checkpoints/epoch\=282.ckpt" \
    scenario_builder=nuplan \
    scenario_filter=val14_split \
    number_of_gpus_allocated_per_simulation=1.0 \
    experiment_name=evaluations_diffusion_xy_64 \
    worker.threads_per_node=4