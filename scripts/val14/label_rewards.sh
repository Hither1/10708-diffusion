#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_open_loop_model \
    planner=log_future_with_features_planner \
    ego_controller=log_play_back_controller \
    scenario_builder=nuplan \
    scenario_filter=train150k_split_mini \
    number_of_gpus_allocated_per_simulation=0.0 \
    save_samples_to_disk=True \
    save_samples_path="/home/scratch/brianyan/test_online_cache/" \
    worker=single_machine_thread_pool


    # worker.threads_per_node=16

    # planner=ml_planner \
    # planner.ml_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/augmentations/unimodal/2023.08.29.18.32.21/best_model/epoch\=107-step\=108323.ckpt" \