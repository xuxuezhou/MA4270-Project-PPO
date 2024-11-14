# backbone.py
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.fc(x)
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value

    def get_action(self, state):
        logits, value = self.forward(state)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        return action, policy_dist.log_prob(action), value

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

def ppo_update_clipped(actor_critic, optimizer, states, actions, log_probs, returns, advantages, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01):
    new_logits, new_values = actor_critic(states)
    policy_dist = Categorical(logits=new_logits)
    new_log_probs = policy_dist.log_prob(actions)
    entropy = policy_dist.entropy().mean()

    ratio = torch.exp(new_log_probs - log_probs.detach())
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = value_coef * (returns - new_values.squeeze()).pow(2).mean()
    loss = policy_loss + value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def ppo_update_kl(actor_critic, optimizer, states, actions, log_probs, returns, advantages, kl_target=0.01, beta=0.5):
    new_logits, new_values = actor_critic(states)
    policy_dist = Categorical(logits=new_logits)
    new_log_probs = policy_dist.log_prob(actions)

    kl_divergence = (log_probs - new_log_probs).mean()
    ratio = torch.exp(new_log_probs - log_probs.detach())
    policy_loss = -(ratio * advantages).mean()

    value_loss = (returns - new_values.squeeze()).pow(2).mean()
    total_loss = policy_loss + value_loss

    if kl_divergence > 1.5 * kl_target:
        beta *= 2
    elif kl_divergence < kl_target / 1.5:
        beta /= 2

    loss = total_loss + beta * kl_divergence

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def conjugate_gradient(Ax, b, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = r.dot(r)

    for _ in range(nsteps):
        z = Ax(p)
        alpha = rdotr / p.dot(z)
        x += alpha * p
        r -= alpha * z
        new_rdotr = r.dot(r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

def trpo_update(actor_critic, states, actions, advantages, max_kl=0.01, cg_iters=10):
    with torch.no_grad():
        old_logits, _ = actor_critic(states)
        old_dist = Categorical(logits=old_logits)
        old_log_probs = old_dist.log_prob(actions)

    def get_loss_and_kl():
        new_logits, _ = actor_critic(states)
        new_dist = Categorical(logits=new_logits)
        new_log_probs = new_dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr_loss = -(ratio * advantages).mean()
        kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
        return surr_loss, kl

    def fisher_vector_product(vector):
        _, kl = get_loss_and_kl()
        kl_grad = torch.autograd.grad(kl, actor_critic.parameters(), create_graph=True)
        kl_grad = torch.cat([g.view(-1) for g in kl_grad])

        kl_v = (kl_grad * vector).sum()
        kl_hvp = torch.autograd.grad(kl_v, actor_critic.parameters())
        kl_hvp = torch.cat([g.contiguous().view(-1) for g in kl_hvp])
        return kl_hvp + 0.1 * vector  # Add damping

    loss, _ = get_loss_and_kl()
    grads = torch.autograd.grad(loss, actor_critic.parameters())
    grads = torch.cat([g.view(-1) for g in grads]).detach()

    # Solve for step direction
    step_dir = conjugate_gradient(fisher_vector_product, grads, cg_iters)

    # Perform line search to ensure KL divergence constraint is satisfied
    step_size = (2 * max_kl / (step_dir.dot(fisher_vector_product(step_dir)) + 1e-8))**0.5
    old_params = torch.cat([param.data.view(-1) for param in actor_critic.parameters()])

    for step_frac in [0.5**i for i in range(10)]:
        new_params = old_params + step_frac * step_size * step_dir
        offset = 0
        for param in actor_critic.parameters():
            param_length = param.numel()
            param.data.copy_(new_params[offset:offset + param_length].view_as(param))
            offset += param_length

        # Check if KL constraint is satisfied
        _, kl = get_loss_and_kl()
        if kl <= max_kl:
            break

    print(f"TRPO update: Step size {step_size}, Final KL divergence {kl.item()}")

def actor_critic_update(actor_critic, optimizer, states, log_probs, rewards, gamma=0.99):
    values = actor_critic(states)[1]
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    policy_loss = -torch.mean(torch.stack([log_prob * (ret - val.item()) for log_prob, ret, val in zip(log_probs, returns, values)]))
    value_loss = torch.mean((returns - values.squeeze())**2)
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def reinforce_update(actor_critic, optimizer, states, log_probs, rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    policy_loss = -torch.mean(torch.stack([log_prob * ret for log_prob, ret in zip(log_probs, returns)]))
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
