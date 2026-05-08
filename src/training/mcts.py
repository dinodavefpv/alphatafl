import numpy as np
import torch
import math

class MCTSNode:
    def __init__(self, state, parent=None, prior=0):
        self.state = state
        self.parent = parent
        self.children = {} # action_index -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.is_expanded = False
        self.child_priors = None

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_action = -1
        
        # We need to consider all actions with non-zero priors
        for action, prob in enumerate(self.child_priors):
            if prob == 0: continue
            
            if action in self.children:
                child = self.children[action]
                score = child.value + c_puct * child.prior * (math.sqrt(self.visit_count) / (1 + child.visit_count))
            else:
                # Unvisited child
                score = c_puct * prob * (math.sqrt(self.visit_count + 1e-8) / 1) # N(s,a) = 0
                
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def expand(self, action_probs):
        self.is_expanded = True
        self.child_priors = action_probs

class MCTS:
    def __init__(self, model, c_puct=1.4):
        self.model = model
        self.device = next(model.parameters()).device
        self.c_puct = c_puct

    def search(self, state, num_simulations):
        from src.training.utils import state_to_tensor, get_legal_moves_mask
        from src.model.network import get_move_from_index
        import alphatafl_engine as engine

        # We need a fresh state because we apply moves
        # But for root, we use the provided state.
        root = MCTSNode(state)
        
        # Initial expansion of root
        with torch.no_grad():
            tensor = state_to_tensor(state).unsqueeze(0).to(self.device)
            policy, value = self.model(tensor)
            policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
            mask = get_legal_moves_mask(state).cpu().numpy()
            policy *= mask # Mask illegal moves
            if policy.sum() > 0:
                policy /= policy.sum()
            
            root.expand(policy)

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            # Selection
            while node.is_expanded:
                action = node.select_child(self.c_puct)
                if action in node.children:
                    node = node.children[action]
                    search_path.append(node)
                else:
                    # Leaf reached, will expand this action
                    break

            # If node is expanded, it means we found an unvisited child action
            if node.is_expanded:
                # 'action' is the unvisited child
                from_r, from_c, to_r, to_c = get_move_from_index(action)
                new_state = node.state.clone()
                new_state.apply_move(engine.Move(from_r, from_c, to_r, to_c))
                
                child = MCTSNode(new_state, parent=node, prior=node.child_priors[action])
                node.children[action] = child
                node = child
                search_path.append(node)

            # Evaluation and Expansion
            if node.state.winner == engine.Player.NONE:
                with torch.no_grad():
                    tensor = state_to_tensor(node.state).unsqueeze(0).to(self.device)
                    policy, value = self.model(tensor)
                    policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                    mask = get_legal_moves_mask(node.state).cpu().numpy()
                    policy *= mask
                    if policy.sum() > 0:
                        policy /= policy.sum()
                    
                    value = value.item()
                    node.expand(policy)
            else:
                # Terminal state
                if node.state.winner == engine.Player.ATTACKER:
                    value = 1.0
                elif node.state.winner == engine.Player.DEFENDER:
                    value = -1.0
                else:
                    value = 0.0
                
                # Turn z into perspective of the player about to move in this node
                # If winner is ATTACKER and it's currently ATTACKER's turn, then v = 1.0
                # Wait, if someone just moved and won, it's now the OTHER player's turn.
                # So if ATTACKER just moved and won, current_turn is DEFENDER.
                # v for DEFENDER should be -1.0.
                if node.state.current_turn == engine.Player.DEFENDER:
                    # z was 1.0 for Attacker win, so v = -1.0 for Defender
                    value = -value if node.state.winner != engine.Player.NONE else 0.0
                else:
                    # current_turn is ATTACKER, so v = 1.0 for Attacker win
                    pass
                # Actually, simpler: winner is always absolute. 
                # value = 1 if current_turn wins, -1 if current_turn loses.
                # But my 'value' is absolute (1 for Attacker).
                # Let's fix this.
                if node.state.winner == engine.Player.ATTACKER:
                    value = 1.0 if node.state.current_turn == engine.Player.ATTACKER else -1.0
                elif node.state.winner == engine.Player.DEFENDER:
                    value = 1.0 if node.state.current_turn == engine.Player.DEFENDER else -1.0
                else:
                    value = 0.0

            # Backup
            curr_value = value
            for n in reversed(search_path):
                n.value_sum += curr_value
                n.visit_count += 1
                curr_value = -curr_value

        # Return visit counts as policy
        probs = np.zeros(11 * 11 * 40)
        for action, child in root.children.items():
            if child:
                probs[action] = child.visit_count
        
        probs /= probs.sum()
        return probs
