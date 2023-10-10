#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_open_loop_model \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion-v3-mini/2023.08.21.12.10.10/checkpoints/epoch\=200.ckpt" \
    planner.ml_planner.replan_freq=5 \
    scenario_builder=nuplan \
    scenario_filter=reduced_val14_split \
    number_of_gpus_allocated_per_simulation=0.25 \
    experiment_name=diffusion_rel \
    worker.threads_per_node=8 \
    planner.ml_planner.dump_gifs=False
