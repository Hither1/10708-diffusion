#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_open_loop_model \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/unimodal/2023.05.24.17.09.08/checkpoints/epoch\=99.ckpt" \
    scenario_builder=nuplan \
    scenario_filter=val14_split \
    number_of_gpus_allocated_per_simulation=0.25 \
    experiment_name=evaluations_unimodal \
    worker.threads_per_node=16