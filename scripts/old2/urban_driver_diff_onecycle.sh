#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=SKRRRT \
    job_name=diffusion_onecyclelr \
    py_func=train \
    +training=training_urban_driver_diffusion_model \
    cache.cache_path=/home/extra_scratch/brianyan/cache_urban_driver \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    scenario_builder=nuplan \
    lightning.trainer.params.max_epochs=450 \
    data_loader.params.batch_size=512 \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_val_batches=5 \
    lightning.trainer.params.check_val_every_n_epoch=5 \
    optimizer=adamw \
    optimizer.lr=0.001 \
    lr_scheduler=one_cycle_lr \
    data_augmentation.kinematic_history_generic_agent_augmentation.use_original=False