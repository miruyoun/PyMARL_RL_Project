import json
import matplotlib.pyplot as plt
import numpy as np

# Run metadata
runs = [
    {"id": "8m_relaxed", "label": "8m"},
    {"id": "2s3z_relaxed", "label": "2s3z"},
    {"id": "MMM2_relaxed", "label": "MMM2"},
    {"id": "corridor_relaxed", "label": "corridor"},
]

base_path = "/projectnb/ds543/miruyoun/pymarl/results/sacred"
plt.figure(figsize=(6, 3))

for run in runs:
    run_id = run["id"]
    label = run["label"]
    json_path = f"{base_path}/{run_id}/info.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    # IMPORTANT: Look for the correct key for loss
    if "loss_T" not in data or "loss" not in data:
        print(f"Warning: {run_id} does not have loss data.")
        continue

    x = np.array(data["loss_T"]) / 1_000_000  # millions of steps
    y = np.array(data["loss"])

    plt.plot(x, y, label=label)

# Plot formatting
plt.xlabel("Timesteps (Millions)")
plt.ylabel("Loss")
plt.title("Training Loss Over Time (RelaxedQMIX)")
plt.legend(title="Map", loc="upper right", fontsize = 8)
plt.grid(True)
plt.tight_layout()
plt.savefig("RelaxedQMIX_Loss.png", dpi=300)
# plt.show()  # Uncomment if running interactively
