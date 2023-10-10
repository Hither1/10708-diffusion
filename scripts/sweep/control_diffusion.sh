#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    model=urban_driver_diffusion_model \
    planner=diffusion_planner \
    planner.diffusion_planner.checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion/2023.08.16.13.05.18/checkpoints/epoch\=90.ckpt" \
    scenario_builder=nuplan \
    scenario_filter=one_hand_picked_scenario \
    number_of_gpus_allocated_per_simulation=1.0 \
    worker=sequential \
    experiment_name=control_test \
    model.predictions_per_sample=16 \
    planner.diffusion_planner.constraint_mode=ours \
    planner.diffusion_planner.goal_mode=llm

# /zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion/2023.08.11.14.50.04/checkpoints/epoch\=44.ckpt
# /zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion/2023.05.24.15.27.00/checkpoints/epoch\=91.ckpt
# /zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/diffusion/2023.07.04.14.40.09/checkpoints/epoch\=282.ckpt
# /home/scratch/brianyan/nuplan_exp/exp/diffusion/conditional_dxdy_full_4gpu/2023.07.04.14.39.51/checkpoints/epoch\=143.ckpt
# /home/scratch/brianyan/nuplan_exp/exp/diffusion/conditional_dxdy_v2/2023.06.30.13.33.00/checkpoints/epoch\=333.ckpt
# /home/scratch/brianyan/nuplan_exp/exp/diffusion/unconditional_dxdy/2023.06.30.13.20.32/checkpoints/epoch\=328.ckpt
# /home/scratch/brianyan/nuplan_exp/exp/diffusion/conditional_xy/2023.06.28.16.24.33/checkpoints/epoch\=499.ckpt
# /home/scratch/brianyan/nuplan_exp/exp/diffusion/unconditional_xy/2023.06.28.16.24.39/checkpoints/epoch\=499.ckpt
# /home/scratch/brianyan/nuplan_exp/exp/diffusion/unconditional_dxdy/2023.06.28.16.24.40/checkpoints/epoch\=125.ckpt
# "/zfsauton/datasets/ArgoRL/brianyan/2023.06.04.14.58.25/checkpoints/epoch\=6.ckpt"
# /home/scratch/brianyan/nuplan_exp/exp/diffusion/unconditional_dxdy/2023.06.28.16.24.40/checkpoints/epoch\=125.ckpt
# /home/scratch/brianyan/nuplan_exp/exp/diffusion/unconditional_xy/2023.06.28.16.24.39/checkpoints/epoch\=125.ckpt