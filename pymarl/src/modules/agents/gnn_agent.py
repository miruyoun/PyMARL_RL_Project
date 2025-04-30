import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNAgent(nn.Module):
    """
    Simple GNN agent for PyMARL that can be used with QMIX
    """
    def __init__(self, input_shape, args):
        super(GNNAgent, self).__init__()
        self.args = args
        
        # Initial feature transformation
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        
        # Graph convolution layer
        self.graph_conv = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        # Output layer
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        
    def init_hidden(self):
        # For compatibility with PyMARL's expected interface
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        
    def forward(self, inputs, hidden_state, adj_matrix=None):
        batch_size = inputs.size(0)
        
        # Process individual features
        x = F.relu(self.fc1(inputs))
        
        # If no adjacency matrix provided, skip graph operations
        if adj_matrix is None:
            h = x
        else:
            # Reshape for graph operations if needed
            n_agents = self.args.n_agents
            if len(x.shape) == 2:  # [batch_size*n_agents, hidden_dim]
                x_graph = x.view(batch_size, n_agents, -1)
            else:
                x_graph = x
            
            # Apply weights to features
            h_w = self.graph_conv(x_graph)
            
            # Normalize adjacency matrix
            row_sum = adj_matrix.sum(dim=-1, keepdim=True)
            norm_adj = adj_matrix / (row_sum + 1e-10)
            
            # Message passing
            h_graph = torch.bmm(norm_adj, h_w)
            h_graph = F.relu(h_graph)
            
            # Reshape back if needed
            if len(x.shape) == 2:  # [batch_size*n_agents, hidden_dim]
                h = h_graph.view(batch_size * n_agents, -1)
            else:
                h = h_graph
        
        # Output layer
        q = self.fc2(h)
        
        return q, hidden_state