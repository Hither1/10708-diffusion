#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=leggo \
    job_name=behavior_transformer \
    py_func=train \
    +training=training_behavior_transformer \
    cache.cache_path=/home/scratch/brianyan/cache_urban_driver_v2 \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    scenario_builder=nuplan \
    lightning.trainer.params.max_epochs=400 \
    data_loader.params.batch_size=64 \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_val_batches=20 \
    lightning.trainer.params.check_val_every_n_epoch=1 \
    callbacks.visualization_callback.frequency=5 \
    optimizer=adamw \
    optimizer.lr=0.0001 \
    warm_up_lr_scheduler=linear_warm_up \
    lr_scheduler=multistep_lr \
    data_augmentation.kinematic_history_generic_agent_augmentation.use_original=False