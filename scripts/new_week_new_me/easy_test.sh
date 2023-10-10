#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=easy \
    job_name=diffusion \
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
    lightning.trainer.params.limit_val_batches=10 \
    lightning.trainer.params.check_val_every_n_epoch=1 \
    callbacks.model_checkpoint_callback.every_n_epochs=10000 \
    optimizer=adamw \
    optimizer.lr=1e-4 \
    callbacks.visualization_callback.skip_train=True \
    model.unconditional=False \
    model.use_deltas=True \
    model.max_dist=25 \
    model.use_relative=True \
    model.noise_scheduler=edm \
    model.use_noise_token=True \
    model.predictions_per_sample=16 \
    model.easy_validation=True