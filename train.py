from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure
from environment import create_environment
from datetime import datetime

def train_agent(env, total_timesteps=100000, learning_rate=3e-4, n_steps=1024, batch_size=64):
    """
    Train a PPO agent on the MiniGrid environment.
    Returns the trained model and environment.
    """

    # Configure training parameters
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set up callbacks
    eval_env = create_environment()  # Separate environment for evaluation
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=10.0,  # Stop when reward exceeds this value
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path="./logs/best_model",
        log_path="./logs",
        eval_freq=10000,
        deterministic=True,
    )

    # Initialize logger
    logger = configure("./logs", ["stdout", "csv", "tensorboard"])
    logger.record("experiment_details/learning_rate", learning_rate)
    logger.record("experiment_details/n_steps", n_steps)
    logger.record("experiment_details/batch_size", batch_size)
    logger.record("experiment_details/timestamp", timestamp)

    # Initialize the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        tensorboard_log="./logs/tensorboard",
    )

    model.set_logger(logger)  # Attach the logger

    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
    )
    print("Training complete.")

    # Save the trained model with a timestamp
    model_name = f"ppo_minigrid_{timestamp}.zip"
    model.save(model_name)

    # Save the model name to a file for later use
    with open("last_model_name.txt", "w") as f:
        f.write(model_name)
        
    print(f"Model saved as {model_name}")

    return model

if __name__ == "__main__":
    # Train the agent using the default parameters
    trained_model, training_env = train_agent()

    # Test the trained agent
    print("Testing trained agent...")
    obs = training_env.reset()
    for _ in range(1000):  # Test for 1000 steps
        action, _ = trained_model.predict(obs)
        obs, reward, done, info = training_env.step(action)
        training_env.render()
        if done:
            obs = training_env.reset()
    training_env.close()
