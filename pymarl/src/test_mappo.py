# src/test_mappo.py

import torch
import argparse
from runners.on_policy_runner import OnPolicyRunner
from controllers.mappo_controller import MAPPOController
from learners.mappo_learner import MAPPOLearner
from smac.env import StarCraft2Env  # Assuming you use SMAC

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--map_name", type=str, default="MMM2")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--max_obs_len", type=int, default=2000)
    parser.add_argument("--max_state_len", type=int, default=322)  # ✅ only once
    return parser.parse_args()

def main():
    args = get_args()

    # Create Environment
    env_args = {
        "map_name": args.map_name,
        "difficulty": "7",  # easy normal difficulty
        "reward_only_positive": False,
        "reward_scale": True,
    }
    env = StarCraft2Env(**env_args)
    env_info = env.get_env_info()

    # Scheme (observation, action, state shapes)
    scheme = {
        "obs": {"vshape": env_info["obs_shape"]},
        "state": {"vshape": env_info["state_shape"]},
        "actions": {"vshape": env_info["n_actions"]},
    }
    groups = None

    # Setup MAC and Learner
    mac = MAPPOController(scheme, groups, args)
    learner = MAPPOLearner(mac, args)
    runner = OnPolicyRunner(env, mac, args)

    # Initialize Environment
    env.reset()

    print("Running 1 rollout...")
    batch, episode_reward = runner.run()
    print(f"Batch keys: {batch.keys()}")
    print(f"Episode reward: {episode_reward}")

    print("Training 1 step...")
    learner.train(batch)
    print("Training step completed successfully!")

    env.close()

if __name__ == "__main__":
    main()

