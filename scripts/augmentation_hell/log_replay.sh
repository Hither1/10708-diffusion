#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_open_loop_model \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/augmentations/unimodal/2023.08.29.18.32.21/best_model/epoch\=107-step\=108323.ckpt" \
    scenario_builder=nuplan \
    scenario_filter=one_of_each_scenario_type \
    number_of_gpus_allocated_per_simulation=0.25 \
    planner.ml_planner.dump_gifs=True \
    planner.ml_planner.dump_gifs_path="/zfsauton2/home/brianyan/nuplan-devkit/nuplan/planning/simulation/planner/ml_planner/viz/log_replay/" \
    planner.ml_planner.log_replay=True \
    worker.threads_per_node=16