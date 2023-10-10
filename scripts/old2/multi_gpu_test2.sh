#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=multi_gpu_test \
    job_name=1gpu \
    py_func=val \
    +training=training_urban_driver_diffusion_model \
    cache.cache_path=/home/scratch/brianyan/cache_urban_driver_v2 \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    scenario_builder=nuplan \
    lightning.trainer.params.max_epochs=1 \
    data_loader.params.batch_size=64 \
    data_loader.val_params.batch_size=8 \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_train_batches=2 \
    lightning.trainer.params.limit_val_batches=5 \
    optimizer=adamw \
    optimizer.lr=0.0001 \
    model.load_checkpoint_path="/home/scratch/brianyan/nuplan_exp/exp/diffusion_v2/8gpu/2023.05.30.16.47.39/checkpoints/epoch\=79.ckpt" \
    data_augmentation.kinematic_history_generic_agent_augmentation.use_original=False