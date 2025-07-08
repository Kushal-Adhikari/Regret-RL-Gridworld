# visualize.py - Plot training metrics using saved logs

from utils import load_logs_npz, plot_training_curves, plot_regret_heatmap
import numpy as np

# -----------------------------
# Load Logs
# -----------------------------
logs = load_logs_npz("logs.npz")

# -----------------------------
# Plot Curves
# -----------------------------
plot_training_curves(logs)

# -----------------------------
# Optional: Load and Plot Heatmap
# -----------------------------
try:
    regret_map = np.load("regret_map.npy")
    plot_regret_heatmap(regret_map)
except FileNotFoundError:
    print("No regret_map.npy found. Skipping heatmap.")
