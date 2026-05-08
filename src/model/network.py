import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AlphaTaflNet(nn.Module):
    def __init__(self, board_size=11, in_channels=14, num_res_blocks=10, num_filters=128):
        super(AlphaTaflNet, self).__init__()
        self.board_size = board_size
        
        # Initial Conv Block
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        # Residual Tower
        self.res_blocks = nn.ModuleList([ResBlock(num_filters) for _ in range(num_res_blocks)])
        
        # Policy Head
        self.policy_head = nn.Sequential(
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, 40, kernel_size=1, bias=True)
        )
        
        # Value Head
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value_fc1 = nn.Linear(board_size * board_size, num_filters)
        self.value_fc2 = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = self.conv_block(x)
        for block in self.res_blocks:
            x = block(x)
        
        # Policy
        p = self.policy_head(x)
        p = p.permute(0, 2, 3, 1)
        p = p.reshape(p.size(0), -1)
        # Note: We'll apply log_softmax during training or softmax during inference
        
        # Value
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v

def get_action_index(from_r, from_c, to_r, to_c, board_size=11):
    # Determine direction and distance
    if to_r == from_r: # Horizontal
        dist = to_c - from_c
        if dist > 0: # Right
            action_type = 0 # 0-9: Right 1-10
            action_val = dist - 1
        else: # Left
            action_type = 1 # 10-19: Left 1-10
            action_val = abs(dist) - 1
    else: # Vertical
        dist = to_r - from_r
        if dist > 0: # Down
            action_type = 2 # 20-29: Down 1-10
            action_val = dist - 1
        else: # Up
            action_type = 3 # 30-39: Up 1-10
            action_val = abs(dist) - 1
            
    return (from_r * board_size + from_c) * 40 + (action_type * 10 + action_val)

def get_move_from_index(index, board_size=11):
    piece_idx = index // 40
    action_idx = index % 40
    
    from_r = piece_idx // board_size
    from_c = piece_idx % board_size
    
    action_type = action_idx // 10
    dist = (action_idx % 10) + 1
    
    if action_type == 0: # Right
        to_r, to_c = from_r, from_c + dist
    elif action_type == 1: # Left
        to_r, to_c = from_r, from_c - dist
    elif action_type == 2: # Down
        to_r, to_c = from_r + dist, from_c
    else: # Up
        to_r, to_c = from_r - dist, from_c
        
    return from_r, from_c, to_r, to_c
