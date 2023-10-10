#!/usr/bin/bash

export HOME=/home/scratch/brianyan/
python nuplan/planning/script/run_training.py \
    experiment_name=diffusion-ma-easy \
    job_name=no_ma \
    py_func=train \
    +training=training_urban_driver_diffusion_model_ma \
    cache.cache_path=/home/extra_scratch/brianyan/cache_urban_driver_ma \
    cache.force_feature_computation=False \
    cache.use_cache_without_dataset=True \
    lightning.trainer.params.max_epochs=500 \
    data_loader.params.batch_size=256 \
    data_loader.val_params.batch_size=4 \
    data_loader.val_params.shuffle=True \
    data_loader.params.num_workers=8 \
    data_loader.params.pin_memory=False \
    lightning.trainer.params.limit_val_batches=10 \
    lightning.trainer.params.check_val_every_n_epoch=1 \
    callbacks.model_checkpoint_callback.every_n_epochs=10 \
    optimizer=adamw \
    optimizer.lr=1e-4 \
    callbacks.visualization_callback.skip_train=True \
    model.predictions_per_sample=16 \
    model.early_map_attention=False \
    model.use_coarse_to_fine_attention=False \
    objective.imitation_objective_ma.weight=0.0 \
    model.easy_validation=True \
    lightning.trainer.checkpoint.resume_training=True