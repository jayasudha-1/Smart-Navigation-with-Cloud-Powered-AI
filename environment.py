import gymnasium
import minigrid
import os
import numpy as np
import time  # For adding delays between frames
from minigrid.wrappers import RGBImgPartialObsWrapper, FullyObsWrapper
from gymnasium import RewardWrapper, ObservationWrapper
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_util import make_vec_env


class CustomObservationWrapper(ObservationWrapper):
    """Custom wrapper to flatten and filter the observation space."""
    def __init__(self, env):
        super().__init__(env)
        # Extract only flattenable parts of the observation space
        flattenable_keys = ['image', 'direction']
        flattenable_spaces = {key: env.observation_space.spaces[key] for key in flattenable_keys}
        flattened_size = int(sum(np.prod(space.shape) if hasattr(space, 'shape') else 1 for space in flattenable_spaces.values()))  # Ensure integer shape
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flattened_size,), dtype=np.float32
        )
    
    def observation(self, observation):
        # Flatten and concatenate the selected keys
        return np.concatenate([
            observation['image'].flatten(),  # Flatten the image
            np.array([observation['direction']], dtype=np.float32)  # Convert direction to an array
        ])

# Check if customenv.py exists in the same directory
if os.path.exists(os.path.join(os.path.dirname(__file__), "customenv.py")):
    from customenv import CustomMazeSoccerBallEnv
else:
    CustomMazeSoccerBallEnv = None


def create_environment():
    """Creates and returns the RL training environment."""
    def make_env():
        if CustomMazeSoccerBallEnv:
            print("Using custom environment from customenv.py.")
            env = CustomMazeSoccerBallEnv(size=25)
            env = FullyObsWrapper(env)  # Ensure fully observable
            env = CustomObservationWrapper(env)  # Use custom observation wrapper
        else:
            print("customenv.py not found. Using fallback environment.")
            env = gymnasium.make('MiniGrid-Dynamic-Obstacles-16x16-v0', render_mode="human")
            env = FullyObsWrapper(env)  # Ensure fully observable
            env = CustomObservationWrapper(env)  # Use custom observation wrapper
        return env

    return make_vec_env(make_env, n_envs=1)

if __name__ == "__main__":
    # Create the environment
    env = create_environment()
    
    # Reset the environment and get the initial observation
    reset_output = env.reset()
    if isinstance(reset_output, tuple):  # Gymnasium >=0.26 returns (obs, info)
        obs, _ = reset_output
    else:  # Older Gym versions return just obs
        obs = reset_output
    
    # Debugging: Print observation details
    print("Custom observation shape:", obs.shape)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    # Simulate one episode
    done = [False]  # Batched format for `done`
    while not all(done):  # Use `all()` to check for completion across all envs
        env.render()  # Render the environment visually
        action = [env.action_space.sample()]  # Wrap action in a list for batch input
        print(f"Action taken: {action}")
        obs, reward, done, info = env.step(action)  # Unpack batched results
        print(f"Reward: {reward}, Done: {done}, Info: {info}")
    
    print("Episode complete.")