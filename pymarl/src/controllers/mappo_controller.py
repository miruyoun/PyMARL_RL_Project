import torch
from modules.agents.mappo_agent import MAPPOAgent

class MAPPOController:
    def __init__(self, scheme, groups, args):
        if isinstance(args.device, str):
            args.device = torch.device(args.device)

        self.args = args
        self.device = args.device

        obs_dim = scheme["obs"]["vshape"]
        if isinstance(obs_dim, tuple):
            obs_dim = obs_dim[0]

        state_dim = scheme["state"]["vshape"]
        if isinstance(state_dim, tuple):
            state_dim = state_dim[0]

        action_dim = scheme["actions"]["vshape"][0]

        self.agent = MAPPOAgent(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            device=self.device
        ).to(self.device)


        for name, param in self.agent.named_parameters():
            assert param.device == self.device, f"Parameter {name} not on device {self.device}"

    def parameters(self):
        return self.agent.parameters()

    def act(self, obs):
        obs = obs.to(self.device)
        return self.agent.act(obs)

    def evaluate_value(self, state):
        state = state.to(self.device)
        return self.agent.evaluate_value(state)

    def select_actions(self, batch, t_ep, t_env, test_mode=False):
        obs = batch["obs"][:, t_ep]
        obs = obs.squeeze(0)
        obs = obs.reshape(-1, obs.shape[-1])
        return self.act(obs)

    def init_hidden(self, batch_size):
        pass
