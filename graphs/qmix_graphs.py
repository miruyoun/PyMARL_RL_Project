import json
import matplotlib.pyplot as plt
import numpy as np

# Run metadata
runs = [
    {"id": "8m_qmix", "label": "8m"},
    {"id": "2s3z_qmix", "label": "2s3z"},
    {"id": "MMM2_qmix", "label": "MMM2"},
    # {"id": 7, "label": "2c_vs_64zg"},
    # { "id": 9, "label": "1c3s5z"},
    # {"id": 10, "label": "10m_vs_11m"},
    {"id": "corridor_qmix", "label": "corridor"},
]

base_path = "/projectnb/ds543/miruyoun/pymarl/results/sacred"
plt.figure(figsize=(6, 6))

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
plt.title("Battle Win Rate Over Training (QMIX)")
plt.legend(title="Map")
plt.grid(True)
plt.tight_layout()
plt.savefig("QMIX.png", dpi=300)
# plt.show()  # Uncomment if using GUI
