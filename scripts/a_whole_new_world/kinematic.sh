#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=kinematic_v2 \
    job_name=kinematic_verlet_ddim_aug \
    py_func=train \
    +training=training_kinematic_diffusion_model \
    cache.cache_path=/home/extra_scratch/brianyan/cache_urban_driver_ma \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    lightning.trainer.params.max_epochs=500 \
    data_loader.params.batch_size=256 \
    data_loader.val_params.batch_size=4 \
    data_loader.val_params.shuffle=True \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_val_batches=10 \
    lightning.trainer.params.check_val_every_n_epoch=1 \
    callbacks.model_checkpoint_callback.every_n_epochs=10 \
    optimizer=adamw \
    optimizer.lr=1e-4 \
    callbacks.visualization_callback.skip_train=False \
    model.T=100
    