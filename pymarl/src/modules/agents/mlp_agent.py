import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        
        print(f"MLPAgent init with input_shape: {input_shape}")
        print(f"Hidden dim: {args.hidden_dim}")
        print(f"Action space: {args.n_actions}")
        
        # Network architecture
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.policy_head = nn.Linear(args.hidden_dim, args.n_actions)
        self.value_head = nn.Linear(args.hidden_dim, 1)
        
        # Print model parameter shapes
        print("Model parameter shapes:")
        print(f"  fc1.weight: {self.fc1.weight.shape}")
        print(f"  fc2.weight: {self.fc2.weight.shape}")
        print(f"  policy_head.weight: {self.policy_head.weight.shape}")
        print(f"  value_head.weight: {self.value_head.weight.shape}")
    
    def forward(self, inputs):
        print(f"MLPAgent forward with inputs shape: {inputs.shape}")
        
        # Safety check for input dimensions
        if inputs.dim() == 3:
            # Flatten batch and agents dimensions for processing
            batch_size, n_agents, feat_dim = inputs.shape
            inputs_reshaped = inputs.reshape(-1, feat_dim)
            print(f"Reshaped inputs: {inputs_reshaped.shape}")
            
            # Check if the feature dimension matches expected input
            if feat_dim != self.input_shape:
                print(f"WARNING: Input feature dim {feat_dim} doesn't match expected {self.input_shape}")
                if feat_dim > self.input_shape:
                    inputs_reshaped = inputs_reshaped[:, :self.input_shape]  # Truncate
                    print(f"Truncated to: {inputs_reshaped.shape}")
                else:
                    # Pad with zeros to match expected shape
                    padding = torch.zeros(batch_size * n_agents, self.input_shape - feat_dim, device=inputs.device)
                    inputs_reshaped = torch.cat([inputs_reshaped, padding], dim=1)
                    print(f"Padded to: {inputs_reshaped.shape}")
            
            # Process through network
            try:
                x = F.relu(self.fc1(inputs_reshaped))
                print(f"After fc1: {x.shape}")
                
                x = F.relu(self.fc2(x))
                print(f"After fc2: {x.shape}")
                
                # Policy output
                policy = self.policy_head(x)
                print(f"Policy output shape: {policy.shape}")
                
                # Value output
                value = self.value_head(x)
                print(f"Value output shape: {value.shape}")
                
                # Reshape back to [batch, agents, features]
                policy = policy.view(batch_size, n_agents, -1)
                value = value.view(batch_size, n_agents, 1)
                
                return policy, value
            except Exception as e:
                print(f"ERROR in forward pass: {e}")
                print(f"Input shape: {inputs.shape}, Expected: {self.input_shape}")
                raise
        else:
            # Input is already flattened or has unexpected dimensions
            print(f"Unexpected input dimensions: {inputs.shape}")
            try:
                x = F.relu(self.fc1(inputs))
                x = F.relu(self.fc2(x))
                policy = self.policy_head(x)
                value = self.value_head(x)
                return policy, value
            except Exception as e:
                print(f"ERROR in forward pass (non-3D input): {e}")
                print(f"Input shape: {inputs.shape}, Expected: {self.input_shape}")
                raise