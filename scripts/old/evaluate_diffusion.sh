#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=diffusion_model \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/home/scratch/brianyan/nuplan_exp/exp/diffusion/augfix_yespca_weightclamp_pretrain/2023.04.07.14.17.48/checkpoints/epoch\=396.ckpt" \
    scenario_builder=nuplan_mini \
    scenario_filter=all_scenarios \
    scenario_filter.num_scenarios_per_type=2