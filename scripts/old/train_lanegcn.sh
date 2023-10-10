#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=lanegcn \
    py_func=train \
    cache.cache_path=/home/scratch/brianyan/nuplan_exp/cache/ \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    +training=training_vector_model \
    scenario_builder=nuplan \
    lightning.trainer.params.max_epochs=600 \
    data_loader.params.batch_size=32 \
    data_loader.params.num_workers=4