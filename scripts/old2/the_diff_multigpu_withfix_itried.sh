#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=diffusion_v2 \
    job_name=8gpu_syncfix \
    py_func=train \
    +training=training_urban_driver_diffusion_model \
    cache.cache_path=/home/scratch/brianyan/cache_urban_driver_v2 \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    scenario_builder=nuplan \
    lightning.trainer.params.max_epochs=500 \
    data_loader.params.batch_size=73 \
    data_loader.val_params.batch_size=8 \
    data_loader.val_params.shuffle=True \
    data_loader.params.num_workers=4 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_val_batches=5 \
    lightning.trainer.params.check_val_every_n_epoch=2 \
    optimizer=adamw \
    optimizer.lr=1e-5 \
    lr_scheduler=multistep_lr \
    data_augmentation.kinematic_history_generic_agent_augmentation.use_original=False \
    callbacks.visualization_callback.frequency=5 \
    model.load_checkpoint_path="/home/scratch/brianyan/nuplan_exp/exp/diffusion_v2/base/2023.05.24.15.27.00/checkpoints/epoch\=91.ckpt"