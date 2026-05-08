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
    """Single-leaf self-play (fallback / GUI / diagnostic path)."""
    state = engine.GameState()
    
    game_history = []
    moves_made = []
    device = next(model.parameters()).device
    cpu_device = torch.device('cpu')
    
    while state.winner == engine.Player.NONE:
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
        probs_list = mcts.search(state, num_simulations)
        probs = np.array(probs_list)
        
        game_history.append((state_to_tensor(state).to(cpu_device), probs, state.current_turn))
        
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
        
    if state.winner == engine.Player.ATTACKER:
        z = 1.0
    elif state.winner == engine.Player.DEFENDER:
        z = -1.0
    else:
        z = 0.0
        
    training_data = []
    for s_tensor, p, turn in game_history:
        z_p = z if turn == engine.Player.ATTACKER else -z
        training_data.append((
            s_tensor, 
            torch.tensor(p, dtype=torch.float32), 
            torch.tensor([z_p], dtype=torch.float32)
        ))
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    winner_str = state.winner.name
    
    if not os.path.exists("data"):
        os.makedirs("data")
    torch.save(training_data, f"data/game_{timestamp}_{winner_str}.pt")
    
    if not os.path.exists("saves"):
        os.makedirs("saves")
    filename = f"saves/game_{timestamp}_{winner_str}.json"
    with open(filename, "w") as f:
        json.dump(moves_made, f)
        
    return game_history, filename


def self_play_game_batched(worker_id, num_simulations, batch_size,
                           inference_queue, response_queue, replay_queue,
                           verbose=True):
    """Batched self-play using Inference Server (K4)."""
    state = engine.GameState()
    
    game_history = []
    moves_made = []
    cpu_device = torch.device('cpu')
    request_counter = [0]
    
    def eval_fn_batched(states_vector):
        """Python callback passed to C++ batched MCTS."""
        if not states_vector:
            return [], []
        
        # 1. batch_to_tensor from C++ → numpy → torch CPU
        tensor = engine.GameState.batch_to_tensor(states_vector)
        tensor = torch.from_numpy(tensor)  # (N, 14, 11, 11)
        
        # 2. Send to inference server
        req_id = request_counter[0]
        request_counter[0] += 1
        inference_queue.put((worker_id, req_id, tensor))
        
        # 3. Wait for response
        _, policies, values = response_queue.get()  # (N, 4840), (N,)
        
        # 4. Apply legal masks and Dirichlet noise (root-only)
        results_policies = []
        results_values = []
        is_root = (request_counter[0] == 1)  # First batch of this search is root
        
        for i, s in enumerate(states_vector):
            mask = s.get_legal_moves_mask()
            p = policies[i] * mask
            
            # Dirichlet noise for root node only
            if is_root:
                num_legal = int(mask.sum())
                if num_legal > 0:
                    noise = np.random.dirichlet([0.3] * num_legal)
                    noise_full = np.zeros(4840, dtype=np.float32)
                    noise_full[mask > 0] = noise
                    p = 0.75 * p + 0.25 * noise_full
            
            # Renormalize
            p_sum = p.sum()
            if p_sum > 0:
                p /= p_sum
            
            results_policies.append(p.tolist())
            results_values.append(float(values[i]))
        
        return results_policies, results_values
    
    # C++ batched search still calls eval_fn for root evaluation,
    # so we provide a wrapper that calls the batched callback with a single state.
    def eval_fn_single(state_to_eval):
        policies, values = eval_fn_batched([state_to_eval])
        return policies[0], values[0]
    
    # Create MCTS with both callbacks
    mcts = engine.MCTS(
        eval_fn=eval_fn_single,
        eval_fn_batched=eval_fn_batched,
        c_puct=1.4
    )
    
    while state.winner == engine.Player.NONE:
        # Run C++ batched MCTS
        probs_list = mcts.search(state, num_simulations, batch_size)
        probs = np.array(probs_list)
        
        # Store state, probs on CPU for the replay buffer
        game_history.append((state_to_tensor(state).to(cpu_device), probs, state.current_turn))
        
        # Select action
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
    
    # Push to replay queue
    for s_tensor, p, turn in game_history:
        z_p = z if turn == engine.Player.ATTACKER else -z
        replay_queue.put((
            s_tensor,
            torch.tensor(p, dtype=torch.float32),
            torch.tensor([z_p], dtype=torch.float32)
        ))
    
    # Also save moves for GUI replay
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    winner_str = state.winner.name
    if not os.path.exists("saves"):
        os.makedirs("saves")
    filename = f"saves/game_{timestamp}_{winner_str}.json"
    with open(filename, "w") as f:
        json.dump(moves_made, f)
    
    return game_history


if __name__ == "__main__":
    model = AlphaTaflNet()
    model.eval()
    history, saved_path = self_play_game(model, num_simulations=20)
    print(f"Collected {len(history)} states and saved game to {saved_path}.")
