# play.py
import torch
import gym
import argparse
import os
from datetime import datetime
from backbone import ActorCritic

def get_latest_checkpoint(task_dir):
    checkpoints = [f for f in os.listdir(task_dir) if f.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {task_dir}")
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(task_dir, x)), reverse=True)
    return os.path.join(task_dir, checkpoints[0])

def play(env_name, algorithm="ppo_clipped", ckpt=None, num_episodes=10, render=False):
    task_dir = f"models/{env_name}"
    
    if ckpt:
        model_path = os.path.join(task_dir, ckpt)
    else:
        model_path = get_latest_checkpoint(task_dir)

    # Check if the model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint '{model_path}' not found.")
    
    print(f"Loading model from {model_path}")
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError("Unsupported action space type")

    actor_critic = ActorCritic(state_dim, action_dim)
    actor_critic.load_state_dict(torch.load(model_path))
    actor_critic.eval()

    total_return = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = actor_critic.get_action(state_tensor)

            state, reward, done, _ = env.step(action.item())
            episode_reward += reward
            if render:
                env.render()

        total_return += episode_reward
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    env.close()
    
    mean_return = total_return / num_episodes
    print(f"Mean Return over {num_episodes} episodes: {mean_return}")
    
    # Log the mean return to log.txt in the model's directory
    log_path = os.path.join(task_dir, "log.txt")
    with open(log_path, "a") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{timestamp} - Algorithm: {algorithm}, Mean Return: {mean_return}\n")
    print(f"Mean return logged to {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play with a trained RL agent.")
    parser.add_argument("--task", type=str, required=True, help="Gym environment name.")
    parser.add_argument("--algo", type=str, default="ppo_clipped", help="Algorithm used for training.")
    parser.add_argument("--ckpt", type=str, default=None, help="Specify the checkpoint filename to load. If not provided, the latest checkpoint will be used.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to play.")
    parser.add_argument("--render", type=bool, default=False, help="Set to True to render the environment during play.")
    args = parser.parse_args()

    play(args.task, algorithm=args.algo, ckpt=args.ckpt, num_episodes=args.num_episodes, render=args.render)
