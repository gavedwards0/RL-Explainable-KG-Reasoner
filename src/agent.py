import torch
import torch.nn as nn
import torch.nn.functional as F

class RLAgent(nn.Module):
    """
    Policy-based RL Agent for KG Reasoning.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super(RLAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

    def find_path(self, start_node: str, target_type: str) -> Tuple[List[str], float]:
        """
        Simulates the agent finding an explainable path.
        """
        print(f"[Agent] Starting reasoning from: {start_node}")
        
        # In a real system, this would be a loop of policy-driven steps
        # For the demo, we show a 'learned' explainable path
        path = ["Disease_X", "Gene_Y", "Protein_Z", "Path_W", "Drug_A"]
        total_reward = 9.7 # (10 - 3*0.1)
        
        return path, total_reward
