REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .gnn_agent import GNNAgent
REGISTRY["gnn"] = GNNAgent

from .gnn_rnn_agent import GNNRNNAgent
REGISTRY["gnn_rnn"] = GNNRNNAgent

from .gat_rnn_agent import SimpleGATLayer
REGISTRY["gat_rnn"] = SimpleGATLayer

from .mlp_agent import MLPAgent
REGISTRY["mlp"] = MLPAgent

from .mappo_agent import MAPPOAgent
REGISTRY["mappo"] = MAPPOAgent