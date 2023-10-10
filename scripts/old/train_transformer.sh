#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=test \
    job_name=test \
    py_func=train \
    cache.cache_path=/home/scratch/brianyan/cache_diffusion_v2/ \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    +training=training_transformer_model \
    scenario_builder=nuplan \
    lightning.trainer.params.max_epochs=600 \
    data_loader.params.batch_size=32 \
    data_loader.params.num_workers=16 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_val_batches=2 \
    worker=sequential