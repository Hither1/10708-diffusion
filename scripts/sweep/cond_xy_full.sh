#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=diffusion_augfix \
    job_name=conditional_xy_full \
    py_func=train \
    +training=training_urban_driver_diffusion_model \
    cache.cache_path=/home/scratch/brianyan/cache_urban_driver_v2 \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    scenario_builder=nuplan \
    lightning.trainer.params.max_epochs=500 \
    data_loader.params.batch_size=512 \
    data_loader.val_params.batch_size=4 \
    data_loader.val_params.shuffle=True \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_val_batches=5 \
    lightning.trainer.params.check_val_every_n_epoch=2 \
    optimizer=adamw \
    optimizer.lr=1e-4 \
    data_augmentation.kinematic_history_generic_agent_augmentation.use_original=False \
    callbacks.visualization_callback.skip_train=True \
    model.unconditional=False \
    model.use_deltas=False \
    model.load_checkpoint_path="/home/scratch/brianyan/nuplan_exp/exp/diffusion/conditional_xy/2023.06.28.16.24.33/checkpoints/epoch\=499.ckpt"
