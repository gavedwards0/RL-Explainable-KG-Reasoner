import networkx as nx
from typing import List, Tuple, Dict

class KGEnv:
    """
    Knowledge Graph Environment for RL Agents.
    """
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._build_mock_graph()
        self.current_node = None

    def _build_mock_graph(self):
        # Sample Biomedical KG structure
        edges = [
            ("Disease_X", "Gene_Y", "associated_with"),
            ("Gene_Y", "Protein_Z", "codes_for"),
            ("Protein_Z", "Path_W", "participates_in"),
            ("Path_W", "Drug_A", "targeted_by"),
            ("Protein_Z", "Drug_B", "inhibited_by")
        ]
        for u, v, r in edges:
            self.graph.add_edge(u, v, relation=r)

    def reset(self, start_node: str):
        self.current_node = start_node
        return self.current_node

    def step(self, action_idx: int) -> Tuple[str, float, bool]:
        """
        Takes an action (choosing an outgoing edge) and returns next state, reward, and done.
        """
        neighbors = list(self.graph.successors(self.current_node))
        if not neighbors:
            return self.current_node, -1.0, True
        
        # Simple mock action selection
        self.current_node = neighbors[action_idx % len(neighbors)]
        
        # Reward if we reach a 'Drug' node
        done = "Drug" in self.current_node
        reward = 10.0 if done else -0.1
        
        return self.current_node, reward, done

    def get_valid_actions(self, node: str) -> List[str]:
        return list(self.graph.successors(node))
