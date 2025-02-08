import os
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor, VecTransposeImage
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.results_plotter import ts2xy, load_results
import numpy as np

# Ensure correct import path
import sys
sys.path.insert(0, "/home/asalvi/code_workspace/Husky_CS_SB3/train/HuskyCP-gym") 
import huskyCP_gym  

# Define paths
tmp_path = "/home/asalvi/code_workspace/tmp/RedRes2/2WsUnTrCb/"  # Log directory
variant = '2WsUnTrCb'  # Model name

# Ensure the directory exists
os.makedirs(tmp_path, exist_ok=True)

# Set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

# Number of additional timesteps to train
additional_timesteps = 11.5e6

# Define function for creating the environment
def make_env(env_id, rank, seed=0):
    """ Create a parallel environment for training """
    def _init():
        port_no = str(23004 + 2 * rank)
        seed = 1 + rank
        env = gym.make(env_id, port=port_no, seed=seed, track_vel=0.75, log_option=0)
        return env
    return _init

# Environment ID
env_id = "huskyCP_gym/HuskyRL-v0"
num_cpu = 16  # Number of parallel environments

# Re-create the environment
env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)], start_method='fork')
env = VecMonitor(env, filename=tmp_path)  # Monitor the environment
env = VecTransposeImage(env, skip=False)

# **Load previous VecNormalize stats if available**
vec_normalize_path = os.path.join(tmp_path, "vecnormalize.pkl")
if os.path.exists(vec_normalize_path):
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = True  # Keep normalizing observations and rewards
    print("Loaded VecNormalize statistics from previous training.")
else:
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=1000.0, gamma=0.99)

# Load the previously trained model
model_path = os.path.join(tmp_path, variant + ".zip")
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}...")
    model = PPO.load(model_path, env=env)
else:
    raise FileNotFoundError(f"Saved model not found at {model_path}")

# Set logger again
model.set_logger(new_logger)

# Define callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=78125, 
    save_path=tmp_path + "checkpoints/", 
    name_prefix=variant,
    save_replay_buffer=True,
    save_vecnormalize=True
)
callback = CallbackList([checkpoint_callback])

# Continue training
print("Continuing training...")
model.learn(total_timesteps=additional_timesteps, callback=callback, progress_bar=True)

# Save updated model and VecNormalize stats
model.save(os.path.join(tmp_path, variant))
env.save(vec_normalize_path)  # Save VecNormalize state

print("Training continued and model saved successfully!")
