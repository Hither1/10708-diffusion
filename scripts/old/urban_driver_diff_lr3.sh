#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=urban_driver_diffusion \
    job_name=lr_sweep_1e-4 \
    py_func=train \
    +training=training_urban_driver_diffusion_model \
    cache.cache_path=/home/scratch/brianyan/cache_urban_driver_v2 \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    scenario_builder=nuplan \
    lightning.trainer.params.max_epochs=60 \
    data_loader.params.batch_size=512 \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_val_batches=5 \
    optimizer=adamw \
    lr_scheduler=one_cycle_lr \
    optimizer.lr=0.0001 \
    data_augmentation.kinematic_history_generic_agent_augmentation.use_original=True