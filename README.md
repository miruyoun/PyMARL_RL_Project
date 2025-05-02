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

4. Set up StarCraft II and SMAC environments following the instructions in [PyMARL](https://github.com/oxwhirl/pymarl#starcraft-ii-setup).

## 🚀 Usage

Train a Relaxed QMIX agent on a SMAC map:
```bash
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=8m

## 🎥 Video Demos

Watch the Relaxed QMIX training results on YouTube:  
[YouTube Playlist](https://www.youtube.com/playlist?list=PLfNwQXb-4EYiBC-Hm0P8xQDTPxbTGFpBp)
