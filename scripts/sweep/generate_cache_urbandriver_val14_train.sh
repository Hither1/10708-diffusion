HOME=/home/scratch/brianyan/ python nuplan/planning/script/run_training.py \
    experiment_name=cache \
    py_func=cache \
    cache.cache_path=/zfsauton/datasets/ArgoRL/brianyan/cache_val14_urban_driver_train/ \
    cache.force_feature_computation=True \
    +training=training_urban_driver_diffusion_model \
    scenario_builder=nuplan \
    scenario_filter=train150k_split \
    worker=single_machine_thread_pool \
    worker.use_process_pool=True