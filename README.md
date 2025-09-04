# Relaxed QMIX for PyMARL

This project implements a variant of QMIX, called **Relaxed QMIX**, within the [PyMARL](https://github.com/oxwhirl/pymarl) framework.  
The goal is to improve cooperative multi-agent reinforcement learning (MARL) performance in complex environments like StarCraft II.

## 📚 Project Description

- **PyMARL** is a widely used framework for deep multi-agent RL research.
- **QMIX** is a value-based method that factorizes joint action-values into individual agent action-values under a monotonicity constraint.
- **Relaxed QMIX** aims to relax some of QMIX's strong assumptions to allow more flexible coordination between agents, potentially improving learning in difficult tasks.

This project includes:
- Modifications to the mixing network architecture
- Custom loss functions
- Experiment configurations for StarCraft II micromanagement (SMAC) environments

## 🏗️ Project Structure
The repository follows the PyMARL framework with additional modifications for Relaxed QMIX:

relaxed_qmix_pymarl/
├── docs/ # Project documentation (e.g., relaxed_qmix.pdf)
├── graphs/ # Training graphs and visualizations
├── logs/ # Log files and experiment outputs
│
├── pymarl/ # Core PyMARL framework
│ ├── __MACOSX/ # System-generated folder (can be ignored)
│ ├── 3rdparty/ # Third-party dependencies
│ ├── docker/ # Docker setup for reproducible runs
│ ├── results/ # Checkpoints, evaluation results
│ ├── src/ # Source code (controllers, learners, modules, etc.)
│ │
│ ├── install_sc2.sh # Script to install StarCraft II
│ ├── run_interactive.sh # Script to run experiments interactively
│ ├── run.sh # Script to launch training jobs
│ ├── requirements.txt # Python dependencies for PyMARL
│ └── LICENSE # PyMARL license
│
├── pymarl_env/ # Python virtual environment (not tracked in Git)
├── replays/ # Saved StarCraft II replays
├── scripts/ # Helper scripts for experiments
├── smac/ # StarCraft Multi-Agent Challenge (SMAC) environment wrapper
├── StarCraftII/ # StarCraft II binary and maps

## ⚙️ Installation

1. Clone the repository:
    ```bash
    git clone git@github.com:miruyoun/relaxed_qmix_pymarl.git
    cd relaxed_qmix_pymarl
    ```

2. Create and activate a virtual environment (recommended):
    ```bash
    python3 -m venv pymarl_env
    source pymarl_env/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up StarCraft II and SMAC environments following the instructions in [PyMARL](https://github.com/oxwhirl/pymarl).

## 🧪 Example Experiments

3 Marines (3m easy map):
```bash
python3 src/main.py --config=relaxed_qmix --env-config=sc2 with env_args.map_name=3m
```

Marine, Marauder, Medivac (MMM2 harder map):
```bash
python3 src/main.py --config=relaxed_qmix --env-config=sc2 with env_args.map_name=3m
```

## 📄 Paper

For a detailed description of Relaxed QMIX and experiments, see the full  
[project paper (PDF)](docs/relaxed_qmix.pdf).

## 🎥 Video Demos

Watch the Relaxed QMIX training results on YouTube:  
[YouTube Playlist](https://www.youtube.com/playlist?list=PLfNwQXb-4EYiBC-Hm0P8xQDTPxbTGFpBp)