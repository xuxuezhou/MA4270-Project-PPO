# train.py
import torch
import gym
import argparse
import os
from datetime import datetime
from backbone import ActorCritic, compute_gae, ppo_update_clipped, ppo_update_kl, trpo_update, actor_critic_update, reinforce_update

def train(env_name, algorithm="ppo_clipped", num_episodes=1000, gamma=0.99, lam=0.95, batch_size=64):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError("Unsupported action space type")

    actor_critic = ActorCritic(state_dim, action_dim)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)

    # Create a directory for the task if it doesn't exist
    task_dir = f"models/{env_name}"
    os.makedirs(task_dir, exist_ok=True)

    for episode in range(num_episodes):
        states, actions, rewards, log_probs, values = [], [], [], [], []
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = actor_critic.get_action(state_tensor)

            next_state, reward, done, _ = env.step(action.item())
            episode_reward += reward

            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value.squeeze().item())

            state = next_state

        if algorithm == "ppo_clipped":
            values.append(0)
            advantages = compute_gae(rewards, values, gamma, lam)
            returns = [adv + val for adv, val in zip(advantages, values[:-1])]

            states = torch.cat(states)
            actions = torch.cat(actions)
            log_probs = torch.cat(log_probs)
            returns = torch.FloatTensor(returns)
            advantages = torch.FloatTensor(advantages)

            ppo_update_clipped(actor_critic, optimizer, states, actions, log_probs, returns, advantages)
        elif algorithm == "ppo_kl":
            values.append(0)
            advantages = compute_gae(rewards, values, gamma, lam)
            returns = [adv + val for adv, val in zip(advantages, values[:-1])]

            states = torch.cat(states)
            actions = torch.cat(actions)
            log_probs = torch.cat(log_probs)
            returns = torch.FloatTensor(returns)
            advantages = torch.FloatTensor(advantages)

            ppo_update_kl(actor_critic, optimizer, states, actions, log_probs, returns, advantages)
        elif algorithm == "trpo":
            values.append(0)
            advantages = compute_gae(rewards, values, gamma, lam)
            returns = [adv + val for adv, val in zip(advantages, values[:-1])]

            states = torch.cat(states)
            actions = torch.cat(actions)
            advantages = torch.FloatTensor(advantages)

            trpo_update(actor_critic, states, actions, advantages)
        elif algorithm == "actor_critic":
            actor_critic_update(actor_critic, optimizer, torch.cat(states), log_probs, rewards, gamma)
        elif algorithm == "reinforce":
            reinforce_update(actor_critic, optimizer, torch.cat(states), log_probs, rewards, gamma)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    # Save the model with a timestamp in the specified directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(task_dir, f"{algorithm}_{timestamp}.pth")
    torch.save(actor_critic.state_dict(), model_path)  # Ensure model_path is used here
    print(f"Model saved as {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent.")
    parser.add_argument("--task", type=str, default="CartPole-v1", help="Gym environment name.")
    parser.add_argument("--algo", type=str, default="ppo_clipped", choices=["ppo_clipped", "ppo_kl", "trpo", "actor_critic", "reinforce"], help="Algorithm to train.")
    args = parser.parse_args()

    train(args.task, algorithm=args.algo)
