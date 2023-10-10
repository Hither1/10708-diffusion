#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=test \
    job_name=test \
    py_func=train \
    +training=training_urban_driver_diffusion_model_ma \
    cache.cache_path=/home/extra_scratch/brianyan/cache_urban_driver_ma \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    scenario_builder=nuplan \
    scenario_filter.limit_total_scenarios=.01 \
    lightning.trainer.params.max_epochs=200 \
    data_loader.params.batch_size=8 \
    data_loader.val_params.batch_size=2 \
    data_loader.params.num_workers=0 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_train_batches=25 \
    lightning.trainer.params.limit_val_batches=5 \
    lightning.trainer.params.check_val_every_n_epoch=10 \
    optimizer=adamw \
    optimizer.lr=0.0001 \
    data_augmentation=agent_history_augmentation \
    data_augmentation.kinematic_history_generic_agent_augmentation.use_original=False