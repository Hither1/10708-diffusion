#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_open_loop_model \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion/2023.05.24.15.27.00/checkpoints/epoch\=91.ckpt" \
    planner.ml_planner.replan_freq=1 \
    scenario_builder=nuplan \
    scenario_filter=reduced_val14_split \
    number_of_gpus_allocated_per_simulation=0.25 \
    experiment_name=replanning_ablation_1 \
    worker.threads_per_node=8

# /zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion/2023.05.24.15.27.00/checkpoints/epoch\=91.ckpt
# /zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion/2023.07.28.16.53.24/checkpoints/epoch\=273.ckpt
# /zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion/2023.07.31.13.12.28/checkpoints/epoch\=70.ckpt