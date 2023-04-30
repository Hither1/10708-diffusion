# 10708-diffusion
As a 10708 PGM project, this repo contains implementations of
- VQVAE
- Discrete Diffusion
- Continuous Diffusion
  -  Linear/Quadratic Noise Schedules
  -  Heun's 2nd Order Method

### File Structure
- models
  -  diffusion: 
     -  `continous_new.py`: contains the two architectures we mentioned in the report
     -  `continuous.py`
     -  `diffusion_utils.py`
     -  `diffusion_transformers.py`
  -  `diffusion_continuous.py`: this is the main class to initialize the trajectory prediction networks
  -  `vqvae.py`: contains VQVAE


### Setting up the Environment
#### Setting up CARLA
```
mkdir carla
cd carla
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.10.1.tar.gz
tar -xf CARLA_0.9.10.1.tar.gz
tar -xf AdditionalMaps_0.9.10.1.tar.gz
rm CARLA_0.9.10.1.tar.gz
rm AdditionalMaps_0.9.10.1.tar.gz
cd ..
```

#### Collecting the Dataset
```
# without display
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./carla/CarlaUE4.sh --world-port=2000 -opengl
```

### Running the Code
Training the model

```
python scripts/leaderboard/wp_IL/train_diffusion_continuous.py gpu=[0,1,2,3]
```



Running the agent
```
python leaderboard/scripts/run_evaluation.py user=$USER experiments=diffusion eval=longest6
```
