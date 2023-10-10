#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=test \
    job_name=test \
    py_func=train \
    +training=training_urban_driver_diffusion_model \
    cache.cache_path=/home/scratch/brianyan/cache_urban_driver \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    scenario_builder=nuplan \
    lightning.trainer.params.max_epochs=500 \
    data_loader.params.batch_size=256 \
    data_loader.val_params.batch_size=4 \
    data_loader.val_params.shuffle=True \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_val_batches=20 \
    lightning.trainer.params.check_val_every_n_epoch=10 \
    callbacks.model_checkpoint_callback.every_n_epochs=10 \
    optimizer=adamw \
    optimizer.lr=1e-4 \
    data_augmentation.kinematic_history_generic_agent_augmentation.use_original=False \
    callbacks.visualization_callback.skip_train=True \
    model.unconditional=True \
    model.use_deltas=True \
    model.use_relative=True \
    model.predictions_per_sample=16