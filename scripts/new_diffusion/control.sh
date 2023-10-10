#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_diffusion_model \
    planner=diffusion_planner \
    planner.diffusion_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion-v4-mini/abs_no_posenc_v2/2023.09.03.16.51.36/best_model/epoch\=97-step\=98293.ckpt" \
    scenario_builder=nuplan \
    scenario_filter=one_hand_picked_scenario \
    number_of_gpus_allocated_per_simulation=1.0 \
    worker=sequential \
    model.predictions_per_sample=16 \
    planner.diffusion_planner.constraint_mode=ours \
    planner.diffusion_planner.goal_mode=llm \
    planner.diffusion_planner.replan_freq=5