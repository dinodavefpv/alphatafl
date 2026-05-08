import json
import os

file_path = r"C:\Users\Administrator\.gemini\tmp\alphatafl\tool-outputs\session-66fd086d-257c-4665-af3c-a9a0a8dd4e07\run_shell_command_run_shell_command_1778129752725_0_bqzz7fq.txt"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    
text = data["output"]

moves = []
for line in text.strip().split('\n'):
    if 'moves' in line:
        parts = line.split('moves ')[1]
        f_str, t_str = parts.split(' -> ')
        f = eval(f_str)
        t = eval(t_str)
        moves.append({"from": [f[0], f[1]], "to": [t[0], t[1]]})

os.makedirs('saves', exist_ok=True)
with open('saves/restored_first_game.json', 'w') as f:
    json.dump(moves, f)

print(f"Saved very first game with {len(moves)} moves.")
