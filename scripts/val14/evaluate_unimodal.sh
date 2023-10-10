#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_open_loop_model \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/unimodal/2023.07.31.13.12.27/checkpoints/epoch\=99.ckpt" \
    scenario_builder=nuplan \
    scenario_filter=one_hand_picked_scenario_v2 \
    number_of_gpus_allocated_per_simulation=0.1 \
    worker=sequential \
    planner.ml_planner.log_replay=True \
    planner.ml_planner.dump_gifs=True \
    planner.ml_planner.dump_gifs_path="/zfsauton2/home/brianyan/nuplan_garage/viz/" \
    ego_controller=log_play_back_controller