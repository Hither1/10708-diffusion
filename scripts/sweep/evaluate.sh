#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_diffusion_model \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/home/scratch/brianyan/nuplan_exp/exp/diffusion/conditional_xy_full/2023.07.04.14.40.09/checkpoints/epoch\=282.ckpt" \
    scenario_builder=nuplan \
    scenario_filter=val14_split \
    number_of_gpus_allocated_per_simulation=0.25 \
    experiment_name=evaluations_diffusion_xy \
    worker.threads_per_node=16