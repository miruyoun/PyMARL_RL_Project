import json
import matplotlib.pyplot as plt
import numpy as np

# Run metadata for QMIX
runs = [
    {"id": "8m_qmix", "label": "8m"},
    {"id": "2s3z_qmix", "label": "2s3z"},
    {"id": "MMM2_qmix", "label": "MMM2"},
    {"id": "corridor_qmix", "label": "corridor"},
]

base_path = "/projectnb/ds543/miruyoun/pymarl/results/sacred"
plt.figure(figsize=(6, 3))

for run in runs:
    run_id = run["id"]
    label = run["label"]
    json_path = f"{base_path}/{run_id}/info.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract return_mean values from nested format
    y = np.array([entry["value"] for entry in data["return_mean"]])
    x = np.array(data["return_mean_T"]) / 1_000_000  # Convert to millions

    plt.plot(x, y, label=label)

# Plot formatting
plt.xlabel("Timesteps (Millions)")
plt.ylabel("Mean Episode Return")
plt.title("Mean Return Over Training (QMIX)")
plt.legend(title="Map", loc="lower right", fontsize=6, frameon=True)
plt.grid(True)
plt.tight_layout()
plt.savefig("ReturnMean_QMIX.png", dpi=300)
# plt.show()
