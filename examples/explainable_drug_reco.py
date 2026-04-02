import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import KGEnv
from src.agent import RLAgent

def main():
    print("--- RL Explainable KG Reasoner: Drug Discovery Demo ---\n")
    
    # Initialize Environment
    env = KGEnv()
    
    # Initialize Agent (dimensions are placeholders for demo)
    agent = RLAgent(state_dim=128, action_dim=10)
    
    # Run the "Reasoning" process
    print("Agent is navigating the Knowledge Graph to find an explanation path...")
    path, reward = agent.find_path(start_node="Disease_X", target_type="Drug")
    
    print("\n--- Reasoning Path Found ---")
    print(" -> ".join(path))
    print(f"\nFinal Reward: {reward}")
    print("Explanation: Disease_X is associated with Gene_Y, which codes for Protein_Z. "
          "Protein_Z participates in Path_W, which is successfully targeted by Drug_A.")

if __name__ == "__main__":
    main()
