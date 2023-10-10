#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=test \
    job_name=test \
    py_func=train \
    +training=training_multi_agent_model \
    cache.cache_path=/home/extra_scratch/brianyan/test_cache \
    cache.force_feature_computation=True \
    scenario_builder=nuplan_mini \
    lightning.trainer.params.max_epochs=450 \
    data_loader.params.batch_size=8 \
    data_loader.params.num_workers=0 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_train_batches=5 \
    lightning.trainer.params.limit_val_batches=2 \
    lightning.trainer.params.check_val_every_n_epoch=5 \
    optimizer=adamw \
    optimizer.lr=0.0001