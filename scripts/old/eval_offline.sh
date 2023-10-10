#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_offline_evaluation.py \
    experiment_name=evals \
    job_name=k4 \
    py_func=train \
    cache.cache_path=/home/scratch/brianyan/cache_diffusion_v2 \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    +training=training_diffusion_model \
    scenario_builder=nuplan \
    lightning.trainer.params.max_epochs=600 \
    data_loader.params.batch_size=4 \
    data_loader.params.num_workers=4 \
    lightning.trainer.params.limit_val_batches=40 \
    model.use_pca=False