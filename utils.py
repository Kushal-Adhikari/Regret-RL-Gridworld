# utils.py - Helper functions for logging and visualization

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Logging Utilities
# -----------------------------
def init_logs():
    return {
        "reward": [],
        "regret": [],
        "epsilon": [],
    }

def update_logs(logs, reward, regret, epsilon):
    logs["reward"].append(reward)
    logs["regret"].append(regret)
    logs["epsilon"].append(epsilon)

def save_logs_npz(logs, filename="logs.npz"):
    np.savez(filename, **logs)

def load_logs_npz(filename="logs.npz"):
    return dict(np.load(filename))

# -----------------------------
# Plotting Utilities
# -----------------------------
def plot_training_curves(logs):
    episodes = np.arange(1, len(logs["reward"]) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(episodes, logs["reward"], label="Reward")
    plt.title("Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(episodes, logs["regret"], label="Regret", color="red")
    plt.title("Average Regret")
    plt.xlabel("Episode")
    plt.ylabel("Regret")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(episodes, logs["epsilon"], label="Epsilon", color="green")
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_regret_heatmap(regret_map):
    plt.figure(figsize=(6, 6))
    plt.imshow(regret_map, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Average Regret")
    plt.title("State-wise Regret Heatmap")
    for i in range(regret_map.shape[0]):
        for j in range(regret_map.shape[1]):
            plt.text(j, i, f"{regret_map[i, j]:.2f}", ha='center', va='center', color='black')
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.show()
