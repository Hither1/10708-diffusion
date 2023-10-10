#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_open_loop_model \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/augmentations/unimodal/2023.08.29.18.32.21/best_model/epoch\=107-step\=108323.ckpt" \
    scenario_builder=nuplan \
    scenario_filter=reduced_train150k_split \
    number_of_gpus_allocated_per_simulation=0.25 \
    experiment_name=unimodal_evaluations_train \
    planner.ml_planner.dump_gifs=False \
    worker.threads_per_node=16