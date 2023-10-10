#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_diffusion_model \
    planner=diffusion_planner \
    planner.diffusion_planner.checkpoint_path="/home/scratch/brianyan/nuplan_exp/exp/diffusion/new_params_lr4/2023.06.22.15.59.48/checkpoints/epoch\=30.ckpt" \
    scenario_builder=nuplan_challenge \
    scenario_filter=u2 \
    number_of_gpus_allocated_per_simulation=1.0 \
    worker=sequential \
    experiment_name=control_test \
    model.predictions_per_sample=16 \
    planner.diffusion_planner.constraint_mode=ours \
    planner.diffusion_planner.goal_mode=llm


# "/zfsauton/datasets/ArgoRL/brianyan/2023.06.04.14.58.25/checkpoints/epoch\=6.ckpt"