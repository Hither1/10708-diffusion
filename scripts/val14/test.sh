#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_open_loop_model \
    planner=idm_planner \
    scenario_builder=nuplan \
    scenario_filter=lane_change_split \
    number_of_gpus_allocated_per_simulation=1.0 \
    save_samples_to_disk=False \
    save_samples_path="/home/scratch/brianyan/test_online_cache/" \
    worker.threads_per_node=16


    # worker.threads_per_node=16

    # planner=ml_planner \
    # planner.ml_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/augmentations/unimodal/2023.08.29.18.32.21/best_model/epoch\=107-step\=108323.ckpt" \