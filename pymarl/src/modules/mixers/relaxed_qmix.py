import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RelaxedQMIX(nn.Module):
    def __init__(self, args):
        super(RelaxedQMIX, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        
        # Control correction magnitude - defaulting to 0.1 if not specified
        self.epsilon = getattr(args, "correction_epsilon", 0.1)
        self.init_scale = getattr(args, "correction_init_scale", 0.001)

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents)
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim)
            )
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for first mixing layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # State-dependent value function for final output
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

        # Correction network
        self.correction_fc1 = nn.Linear(self.n_agents + self.state_dim, self.embed_dim)
        self.correction_fc2 = nn.Linear(self.embed_dim + self.state_dim, self.embed_dim)
        self.correction_out = nn.Linear(self.embed_dim, 1)
        
        # Initialize correction network with small weights
        nn.init.xavier_uniform_(self.correction_fc1.weight, gain=self.init_scale)
        nn.init.xavier_uniform_(self.correction_fc2.weight, gain=self.init_scale)
        nn.init.xavier_uniform_(self.correction_out.weight, gain=self.init_scale)
        nn.init.constant_(self.correction_out.bias, 0)

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # First mixing layer - standard QMIX
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        # Second mixing layer - standard QMIX
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(states).view(-1, 1, 1)
        y = th.bmm(hidden, w_final) + v
        q_tot_main = y.view(bs, -1, 1)

        # Correction network
        agent_qs_flat = agent_qs.view(-1, self.n_agents)
        correction_input = th.cat([agent_qs_flat, states], dim=-1)
        x = F.relu(self.correction_fc1(correction_input))
        x = th.cat([x, states], dim=-1)
        x = F.relu(self.correction_fc2(x))
        q_correction = self.correction_out(x).view(bs, -1, 1)
        
        # Apply correction with scaling factor - ensure same shape as q_tot_main
        q_tot = q_tot_main + self.epsilon * q_correction
        
        # Make sure both returned tensors have the same shape
        return q_tot, q_correction

    def get_regularization_loss(self):
        """
        Compute regularization loss to control correction magnitude
        """
        # L2 regularization on correction network weights
        l2_reg = th.norm(self.correction_fc1.weight) + \
                 th.norm(self.correction_fc2.weight) + \
                 th.norm(self.correction_out.weight)
        
        return l2_reg