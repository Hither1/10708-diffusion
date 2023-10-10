#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=multi_gpu_test \
    job_name=1gpu \
    py_func=val \
    +training=training_urban_driver_open_loop_model \
    cache.cache_path=/home/extra_scratch/brianyan/cache_urban_driver \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    scenario_builder=nuplan \
    lightning.trainer.params.max_epochs=1 \
    data_loader.params.batch_size=2 \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_train_batches=2 \
    lightning.trainer.params.limit_val_batches=2 \
    optimizer=adamw \
    optimizer.lr=0.0001 \
    model.load_checkpoint_path="/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/unimodal/2023.05.23.00.51.06/checkpoints/epoch\=275.ckpt" \
    data_augmentation.kinematic_history_generic_agent_augmentation.use_original=False