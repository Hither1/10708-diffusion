#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=multiagent \
    job_name=test \
    py_func=train \
    +training=training_multi_agent_diffusion_model \
    cache.cache_path=/home/extra_scratch/brianyan/cache_urban_driver_ma \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    scenario_builder=nuplan \
    lightning.trainer.params.max_epochs=500 \
    data_loader.params.batch_size=32 \
    data_loader.val_params.batch_size=4 \
    data_loader.val_params.shuffle=True \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_val_batches=5 \
    lightning.trainer.params.check_val_every_n_epoch=2 \
    optimizer=adamw \
    optimizer.lr=1e-4 \
    callbacks.visualization_callback.skip_train=True \
    model.rotary_pe=True