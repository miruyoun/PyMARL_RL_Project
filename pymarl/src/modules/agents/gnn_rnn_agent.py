import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNRNNAgent(nn.Module):
    """
    GNN+RNN hybrid agent for PyMARL
    Combines graph processing with recurrent memory
    """
    def __init__(self, input_shape, args):
        super(GNNRNNAgent, self).__init__()
        self.args = args
        
        # Initial feature transformation
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        
        # Simple graph convolution layer
        self.graph_conv = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        # RNN layer (GRU Cell)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        # Output layer
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        
        # Flag to control graph operations (set to False initially if debugging)
        self.use_graph_conv = True
        
    def init_hidden(self):
        # Hidden state initialization for RNN
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        
    def forward(self, inputs, hidden_state, adj_matrix=None):
        batch_size = inputs.size(0)
        
        # Initial feature extraction
        x = F.relu(self.fc1(inputs))
        
        # Apply graph convolution if enabled and adjacency matrix provided
        if self.use_graph_conv and adj_matrix is not None:
            try:
                # Get dimensions
                n_agents = self.args.n_agents
                hidden_dim = self.args.rnn_hidden_dim
                
                # Reshape for graph operations if needed
                if len(x.shape) == 2:  # [batch_size*n_agents, hidden_dim]
                    x_graph = x.view(batch_size, n_agents, hidden_dim)
                else:
                    x_graph = x
                
                # Apply graph convolution weights
                h_w = self.graph_conv(x_graph)
                
                # Normalize adjacency matrix
                row_sum = adj_matrix.sum(dim=-1, keepdim=True)
                norm_adj = adj_matrix / (row_sum + 1e-10)
                
                # Message passing
                h_graph = torch.bmm(norm_adj, h_w)
                h_graph = F.relu(h_graph)
                
                # Reshape back if needed
                if len(x.shape) == 2:
                    h = h_graph.view(batch_size * n_agents, hidden_dim)
                else:
                    h = h_graph
                
                # Skip connection (add original features)
                h = h + x
            except Exception as e:
                # Fall back to using just the node features
                h = x
        else:
            # Skip graph operations
            h = x
        
        # Apply RNN - reshape hidden state
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_out = self.rnn(h, h_in)
        
        # Output layer
        q = self.fc2(h_out)
        
        return q, h_out