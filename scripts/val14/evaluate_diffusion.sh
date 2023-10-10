#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_diffusion_model \
    planner=ml_planner \
    planner.ml_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion-v4-mini/abs_v2/2023.09.03.16.51.37/best_model/epoch\=358-step\=360076.ckpt" \
    scenario_builder=nuplan \
    scenario_filter=reduced_val14_split \
    number_of_gpus_allocated_per_simulation=0.25 \
    experiment_name=diffusion_evals \
    worker.threads_per_node=16

# /zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion/2023.07.28.16.53.24/checkpoints/epoch\=273.ckpt
# /zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion/2023.07.31.13.12.28/checkpoints/epoch\=70.ckpt