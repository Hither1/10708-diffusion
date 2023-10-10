#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=unimodal \
    job_name=base \
    py_func=train \
    +training=training_urban_driver_open_loop_model \
    cache.cache_path=/home/scratch/brianyan/cache_urban_driver_v2 \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    scenario_builder=nuplan \
    lightning.trainer.params.max_epochs=100 \
    data_loader.params.batch_size=512 \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_val_batches=20 \
    optimizer=adamw \
    optimizer.lr=0.0001 \
    data_augmentation.kinematic_history_generic_agent_augmentation.use_original=False