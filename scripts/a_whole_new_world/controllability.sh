#!/usr/bin/bash

export HOME=/home/scratch/huangyus/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=kinematic_diffusion_model \
    planner=factorized_diffusion_planner \
    planner.factorized_diffusion_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/kinematic_v2/kinematic_verlet_ddim_aug/2023.10.04.00.14.10/best_model/epoch\=490.ckpt" \
    scenario_builder=nuplan \
    scenario_filter=one_hand_picked_scenario_v6 \
    number_of_gpus_allocated_per_simulation=1.0 \
    worker=sequential \
    planner.factorized_diffusion_planner.replan_freq=5 \
    planner.factorized_diffusion_planner.dump_gifs=True \
    planner.factorized_diffusion_planner.dump_gifs_path="/zfsauton2/home/huangyus/nuplan-diffusion/nuplan/planning/simulation/planner/ml_planner/viz/"