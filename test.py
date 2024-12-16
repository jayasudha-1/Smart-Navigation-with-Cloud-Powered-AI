from stable_baselines3 import PPO  # For loading the trained model
from environment import create_environment  # To create the testing environment
import time  # For adding delays during rendering


def test_agent(model, env, max_steps=3000):
    """
    Test a trained RL agent on the given environment.

    Parameters:
    - model: The trained PPO model.
    - env: The environment to test in.
    - max_steps: Maximum number of steps to run the test.

    Outputs:
    - Renders the environment and prints rewards and actions.
    """
    obs = env.reset()  # Reset the environment
    done = False
    total_reward = 0

    print("Starting test...")
    for step in range(max_steps):
        # Predict the action from the model
        action, _ = model.predict(obs)

        # Take a step in the environment
        obs, reward, done, info = env.step(action)

        # Render the environment
        env.render()

        # Print debug information
        print(f"Step: {step}, Action: {action}, Reward: {reward}")

        # Accumulate rewards
        total_reward += reward

        # Stop testing if the episode ends
        if done:
            print(f"Episode finished after {step + 1} steps with total reward: {total_reward}")
            break

        # Add a delay to visualize the test
        time.sleep(0.1)

    env.close()

if __name__ == "__main__":
    # Load the last saved model
    with open("last_model_name.txt", "r") as f:
        model_name = f.read().strip()
    model = PPO.load(model_name)

    # Create the environment
    env = create_environment()

    # Test the agent
    test_agent(model, env)
