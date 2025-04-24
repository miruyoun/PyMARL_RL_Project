import json
import matplotlib.pyplot as plt
import numpy as np

# Run metadata
runs = [
    {"id": 12, "label": "MMM2"},
    {"id": 13, "label": "10m_vs_11m"},
    {"id": 14, "label": "8m"},
    {"id": 15, "label": "2s3z"},
]

base_path = "/projectnb/ds543/miruyoun/pymarl/results/sacred"
plt.figure(figsize=(12, 6))

for run in runs:
    run_id = run["id"]
    label = run["label"]
    json_path = f"{base_path}/{run_id}/info.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    x = np.array(data["battle_won_mean_T"]) / 1_000_000  # millions of steps
    y = np.array(data["battle_won_mean"]) * 100           # convert to %

    plt.plot(x, y, label=label)

# Plot formatting
plt.xlabel("Timesteps (Millions)")
plt.ylabel("Win Rate (%)")
plt.title("Battle Win Rate Over Training")
plt.legend(title="Map")
plt.grid(True)
plt.tight_layout()
plt.savefig("GNN_QMIX.png", dpi=300)
# plt.show()  # Uncomment if using GUI
