from environment import create_environment
from train import train_agent
from test import test_agent
from stable_baselines3 import PPO

if __name__ == "__main__":
    # Step 1: Create the environment
    env = create_environment()

    # Train the agent
    print("Starting training process...")
    model = train_agent(env=env, total_timesteps=100000)  # Pass the environment and timesteps

    # Step 3: Test the trained agent
    with open("last_model_name.txt", "r") as f:
        model_name = f.read().strip()
    model = PPO.load("ppo_minigrid")  # Load the saved model
    test_agent(model, env)