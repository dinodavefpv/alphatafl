import sys
import os
import torch
import numpy as np
sys.path.append(os.path.join(os.getcwd(), 'build', 'Release'))

import alphatafl_engine as engine
from src.model.network import AlphaTaflNet, get_move_from_index
from src.training.utils import state_to_tensor, get_legal_moves_mask

import json
import datetime

def self_play_game(model, num_simulations=100, verbose=True):
    state = engine.GameState()
    
    game_history = []
    moves_made = []
    device = next(model.parameters()).device
    cpu_device = torch.device('cpu')
    
    while state.winner == engine.Player.NONE:
        # Create a stateful eval_fn for each search to track the root node
        call_count = [0]
        def eval_fn(state_to_eval):
            with torch.no_grad():
                tensor = state_to_tensor(state_to_eval).unsqueeze(0).to(device)
                policy, value = model(tensor)
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                
                mask = get_legal_moves_mask(state_to_eval).cpu().numpy()
                policy *= mask
                sum_prob = policy.sum()
                if sum_prob > 0:
                    policy /= sum_prob
                
                # Add Dirichlet noise to the root node for exploration
                if call_count[0] == 0:
                    num_legal_moves = np.count_nonzero(mask)
                    if num_legal_moves > 0:
                        noise = np.random.dirichlet([0.3] * num_legal_moves)
                        noise_full = np.zeros_like(policy)
                        noise_full[mask > 0] = noise
                        policy = 0.75 * policy + 0.25 * noise_full
                
                call_count[0] += 1
                val = value.item()
                return policy.tolist(), val

        mcts = engine.MCTS(eval_fn, 1.4)
        
        # Run C++ MCTS
        probs_list = mcts.search(state, num_simulations)
        probs = np.array(probs_list)
        
        # Store state, probs on CPU for the replay buffer
        game_history.append((state_to_tensor(state).to(cpu_device), probs, state.current_turn))
        
        # Select action
        # Use temperature = 1 (proportional sampling) for early game to encourage diversity
        if len(game_history) < 30:
            action = np.random.choice(len(probs), p=probs)
        else:
            action = np.argmax(probs)
            
        from_r, from_c, to_r, to_c = get_move_from_index(action)
        moves_made.append({"from": [int(from_r), int(from_c)], "to": [int(to_r), int(to_c)]})
        
        if verbose:
            print(f"Turn {len(game_history)}: {state.current_turn.name} moves ({from_r}, {from_c}) -> ({to_r}, {to_c})")
        state.apply_move(engine.Move(from_r, from_c, to_r, to_c))
        
        if len(game_history) > 200:
            break
            
    if verbose:
        print(f"Game finished! Winner: {state.winner.name}")
        
    # Calculate final outcome (z)
    if state.winner == engine.Player.ATTACKER:
        z = 1.0
    elif state.winner == engine.Player.DEFENDER:
        z = -1.0
    else:
        z = 0.0
        
    training_data = []
    for s_tensor, p, turn in game_history:
        # z from perspective of 'turn'
        z_p = z if turn == engine.Player.ATTACKER else -z
        training_data.append((
            s_tensor, 
            torch.tensor(p, dtype=torch.float32), 
            torch.tensor([z_p], dtype=torch.float32)
        ))
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    winner_str = state.winner.name
    
    # Save training data
    if not os.path.exists("data"):
        os.makedirs("data")
    torch.save(training_data, f"data/game_{timestamp}_{winner_str}.pt")
    
    # Save moves for GUI
    if not os.path.exists("saves"):
        os.makedirs("saves")
    filename = f"saves/game_{timestamp}_{winner_str}.json"
    with open(filename, "w") as f:
        json.dump(moves_made, f)
        
    return game_history, filename

if __name__ == "__main__":
    model = AlphaTaflNet()
    model.eval()
    history, saved_path = self_play_game(model, num_simulations=20)
    print(f"Collected {len(history)} states and saved game to {saved_path}.")
