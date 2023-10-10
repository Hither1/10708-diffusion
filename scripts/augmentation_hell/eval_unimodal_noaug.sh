#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_open_loop_model \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/augmentations/unimodal_noaug/2023.08.29.18.32.41/best_model/epoch\=97-step\=98293.ckpt" \
    scenario_builder=nuplan \
    scenario_filter=one_of_each_scenario_type \
    number_of_gpus_allocated_per_simulation=0.25 \
    experiment_name=aug_unimodal_noaug \
    planner.ml_planner.dump_gifs=True \
    planner.ml_planner.dump_gifs_path="/zfsauton2/home/brianyan/nuplan-devkit/nuplan/planning/simulation/planner/ml_planner/viz/aug_unimodal_noaug/" \
    worker.threads_per_node=4