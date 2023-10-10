#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=augmentations \
    job_name=unimodal \
    py_func=train \
    +training=training_urban_driver_open_loop_model \
    cache.cache_path=/home/scratch/brianyan/cache_urban_driver \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    scenario_builder=nuplan \
    lightning.trainer.params.max_epochs=500 \
    data_loader.params.batch_size=256 \
    data_loader.val_params.batch_size=256 \
    data_loader.val_params.shuffle=True \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_val_batches=10 \
    lightning.trainer.params.check_val_every_n_epoch=2 \
    callbacks.model_checkpoint_callback.every_n_epochs=10 \
    optimizer=adamw \
    optimizer.lr=1e-4 \
    model.feature_params.standardize=False \
    callbacks.visualization_callback.skip_train=True \
    data_augmentation.dumb_augmentation.history_smoothing=False \
    data_augmentation.history_dropout.augment_prob=0.0 \
    lightning.trainer.checkpoint.resume_training=True \