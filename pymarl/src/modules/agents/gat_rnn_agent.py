import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleGATLayer(nn.Module):
    """
    Simplified Graph Attention Layer with single-head attention
    """
    def __init__(self, in_features, out_features, dropout=0.1):
        super(SimpleGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        
        # Single transformation matrix
        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform_(self.W)
        
        # Simple attention mechanism (single scalar)
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, adj_matrix):
        """
        x: Input features [batch_size, n_agents, in_features]
        adj_matrix: Adjacency matrix [batch_size, n_agents, n_agents]
        """
        batch_size, n_agents, _ = x.shape
        
        # Linear transformation
        h = torch.matmul(x.view(-1, self.in_features), self.W).view(batch_size, n_agents, self.out_features)
        
        # Prepare for attention
        a_input = torch.cat([
            h.repeat(1, 1, n_agents).view(batch_size, n_agents, n_agents, self.out_features),
            h.repeat(1, n_agents, 1).view(batch_size, n_agents, n_agents, self.out_features)
        ], dim=3)
        
        # Compute attention coefficients
        e = self.leaky_relu(torch.matmul(a_input.view(-1, 2 * self.out_features), self.a))
        e = e.view(batch_size, n_agents, n_agents)
        
        # Mask attention for non-neighbors
        masked_e = torch.where(adj_matrix > 0, e, -9e15 * torch.ones_like(e))
        attention = F.softmax(masked_e, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention
        h_out = torch.bmm(attention, h)
        
        return h_out

class SimpleGATRNNAgent(nn.Module):
    """
    Simplified GAT-RNN Agent with reduced complexity
    """
    def __init__(self, input_shape, args):
        super(SimpleGATRNNAgent, self).__init__()
        self.args = args
        
        # Reduced dimensions
        self.hidden_dim = 32  # Smaller hidden dimension
        self.dropout = 0.1
        
        # Initial feature transformation
        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        
        # Single GAT layer instead of two
        self.gat = SimpleGATLayer(
            in_features=self.hidden_dim, 
            out_features=self.hidden_dim,
            dropout=self.dropout
        )
        
        # RNN for temporal processing
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        
        # Output layer
        self.fc2 = nn.Linear(self.hidden_dim, args.n_actions)
        
    def init_hidden(self):
        return self.fc1.weight.new(1, self.hidden_dim).zero_()
        
    def forward(self, inputs, hidden_state, adj_matrix=None):
        # Input shape handling
        if inputs.dim() == 2 and inputs.size(0) == self.args.n_agents:
            # Case: inputs shape is [n_agents, features] without batch dimension
            batch_size = 1
            n_agents = self.args.n_agents
            # Add batch dimension
            inputs_reshaped = inputs.unsqueeze(0)  # [1, n_agents, features]
        elif inputs.dim() == 2 and inputs.size(0) % self.args.n_agents == 0:
            # Case: inputs shape is [batch_size*n_agents, features]
            batch_size = inputs.size(0) // self.args.n_agents
            n_agents = self.args.n_agents
            # Reshape to [batch_size, n_agents, features]
            inputs_reshaped = inputs.view(batch_size, n_agents, -1)
        elif inputs.dim() == 3:
            # Case: inputs already have shape [batch_size, n_agents, features]
            batch_size = inputs.size(0)
            n_agents = inputs.size(1)
            inputs_reshaped = inputs
        else:
            raise RuntimeError(f"Unexpected input shape: {inputs.shape} (n_agents={self.args.n_agents})")
        
        # Initial feature transformation
        x = F.relu(self.fc1(inputs_reshaped))
            
        # Create adjacency matrix if none provided
        if adj_matrix is None:
            adj_matrix = torch.ones(batch_size, n_agents, n_agents).to(x.device)
            # Remove self-loops (diagonal)
            adj_matrix = adj_matrix * (1 - torch.eye(n_agents).unsqueeze(0).to(x.device))
        
        # Apply Graph Attention
        h = self.gat(x, adj_matrix)
        h = F.elu(h)
        
        # Add skip connection
        h = h + x
        
        # Reshape for RNN
        h_flat = h.view(batch_size * n_agents, -1)
        
        # Process hidden state
        if hidden_state.dim() == 2 and hidden_state.size(0) == 1:
            h_in = hidden_state.repeat(batch_size * n_agents, 1)
        else:
            h_in = hidden_state.reshape(-1, self.hidden_dim)
        
        # Apply RNN
        h_out = self.rnn(h_flat, h_in)
        
        # Get Q-values
        q = self.fc2(h_out)
        
        return q, h_out