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
    scenario_filter=one_hand_picked_scenario \
    lightning.trainer.params.max_epochs=500 \
    data_loader.params.batch_size=1 \
    data_loader.val_params.batch_size=1 \
    data_loader.val_params.shuffle=True \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=False \
    data_loader.datamodule.load_single_sample=True \
    lightning.trainer.params.limit_val_batches=10 \
    lightning.trainer.params.check_val_every_n_epoch=1 \
    callbacks.model_checkpoint_callback.every_n_epochs=100000000000 \
    optimizer=adamw \
    optimizer.lr=1e-4 \
    lightning.trainer.params.gradient_clip_val=1.0 \
    callbacks.visualization_callback.skip_train=False \
    model.use_deltas=False \
    model.predictions_per_sample=16 \
    data_augmentation.dumb_augmentation.augment_prob=0.0 \
    data_loader.datamodule.load_single_sample=True