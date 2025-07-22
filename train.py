# train.py - Training loop using RegretAgent and TorchGridWorld

from Agent.Regret_agent import Regret_Agent
from env.gridworld import Grid_World
from utils import init_logs, update_logs, save_logs_npz
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch

# -----------------------------
# Hyperparameters
# -----------------------------
batch_size = 64
state_dim = 2
action_dim = 4
hidden_dim = 64
lr = 0.001
episodes = 10000
max_steps = 50
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 0.995

# -----------------------------
# Initialize Environment and Agent
# -----------------------------
env = Grid_World(batch_size= batch_size, size=5)
agent = Regret_Agent(state_dim, action_dim, hidden_dim, lr)

# -----------------------------
# Logging Initialization
# -----------------------------
logs = init_logs()

# -----------------------------
# Training Loop
# -----------------------------
for episode in range(episodes):
    states = env.reset()
    total_reward = 0
    total_regret = 0

    for _ in range(max_steps):
        actions = agent.select_actions(states, epsilon)

        # Ensure actions is a tensor with shape [batch_size]
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)

        if actions.ndim == 0:
            actions = actions.unsqueeze(0)

        next_states, rewards, dones = env.step(actions)

        loss, avg_regret = agent.train_step(states, actions, next_states, rewards, dones)

        total_reward += rewards.sum().item()
        total_regret += avg_regret * batch_size

        states = next_states.clone()
        if dones.all():
            break

    # Log values for this episode
    update_logs(logs, total_reward / batch_size, total_regret / batch_size, epsilon)

    # Epsilon decay
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Print status every 10 episodes
    if episode % 100 == 0:
        print(f"Episode {episode} | Avg Reward: {logs['reward'][-1]:.2f} | Avg Regret: {logs['regret'][-1]:.4f} | Epsilon: {epsilon:.3f}")

# -----------------------------
# Save Logs
# -----------------------------
save_logs_npz(logs, "logs.npz")
