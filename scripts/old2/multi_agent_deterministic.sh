#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=multi_agent \
    job_name=deterministic \
    py_func=train \
    +training=training_multi_agent_model \
    cache.cache_path=/home/extra_scratch/brianyan/cache_urban_driver_ma \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    lightning.trainer.params.max_epochs=450 \
    data_loader.params.batch_size=256 \
    data_loader.params.num_workers=16 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_val_batches=20 \
    optimizer=adamw \
    optimizer.lr=0.0001 \
    warm_up_lr_scheduler=linear_warm_up \
    lr_scheduler=multistep_lr \