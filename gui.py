import pygame
import sys
import os
import json
import torch
import numpy as np
import threading

# Add the build directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'build', 'Release'))
import alphatafl_engine as engine
from src.model.network import AlphaTaflNet
from src.training.utils import state_to_tensor

# --- CONFIGURATION ---
BOARD_WIDTH, BOARD_HEIGHT = 660, 660
ROWS, COLS = 11, 11
SQUARE_SIZE = BOARD_WIDTH // COLS
BOTTOM_BAR_HEIGHT = 240
WIDTH, HEIGHT = BOARD_WIDTH, BOARD_HEIGHT + BOTTOM_BAR_HEIGHT

# Colors
BOARD_COLOR = (222, 184, 135) # Light wood
LINE_COLOR = (139, 69, 19)    # Dark wood
KING_SIDE_COLOR = (40, 40, 40) # Black
ATTACKER_COLOR = (240, 240, 240) # White
KING_CROSS_COLOR = (255, 255, 255) # White
HIGHLIGHT_COLOR = (100, 200, 100, 150) # Transparent green
BUTTON_COLOR = (100, 100, 100)
BUTTON_HOVER_COLOR = (150, 150, 150)
TEXT_COLOR = (255, 255, 255)
CHART_BG_COLOR = (50, 50, 50)
CHART_LINE_COLOR = (0, 255, 0)

# Game Modes
TWO_PLAYER = 0
ONE_PLAYER_VS_AI = 1
REPLAY = 2

class Button:
    def __init__(self, x, y, width, height, text, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        
    def draw(self, win, mouse_pos):
        color = BUTTON_HOVER_COLOR if self.rect.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(win, color, self.rect)
        pygame.draw.rect(win, (0, 0, 0), self.rect, 2)
        text_surf = self.font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        win.blit(text_surf, text_rect)
        
    def is_clicked(self, event, mouse_pos):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(mouse_pos):
                return True
        return False

class HnefataflGUI:
    def __init__(self, mode=TWO_PLAYER, ai_version="AlphaTafl_v0.1"):
        self.game = engine.GameState()
        self.selected_piece = None
        self.mode = mode
        self.ai_version = ai_version
        
        # ML Model for probability
        self.model = AlphaTaflNet()
        self.model.eval()
        
        # UI Elements
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 24)
        font_sm = pygame.font.SysFont('Arial', 18)
        self.btn_load = Button(10, BOARD_HEIGHT + 10, 110, 40, "Load Game", font_sm)
        self.btn_play_ai = Button(125, BOARD_HEIGHT + 10, 110, 40, "Play AI", font_sm)
        self.btn_start = Button(240, BOARD_HEIGHT + 10, 60, 40, "Start", font_sm)
        self.btn_prev = Button(305, BOARD_HEIGHT + 10, 60, 40, "Prev", font_sm)
        self.btn_next = Button(370, BOARD_HEIGHT + 10, 60, 40, "Next", font_sm)
        self.btn_play = Button(435, BOARD_HEIGHT + 10, 60, 40, "Play", font_sm)
        self.btn_ff = Button(500, BOARD_HEIGHT + 10, 150, 40, "Speed: 1.0x", font_sm)
        
        self.human_side = engine.Player.ATTACKER
        self.is_playing = False
        self.showing_side_selection = False
        self.selection_rect_atk = pygame.Rect(WIDTH//2 - 160, HEIGHT//2 - 50, 150, 100)
        self.selection_rect_def = pygame.Rect(WIDTH//2 + 10, HEIGHT//2 - 50, 150, 100)
        
        self.playback_speed = 1.0 # moves per second
        self.last_move_time = 0
        self.replay_moves = []
        self.replay_index = -1
        self.probabilities = [0.0]
        
        # K6: Async AI search
        self.ai_search_thread = None
        self.ai_search_result = None
        self.ai_is_searching = False
        
        # Check if we have a trained model
        model_path = "models/current_best.pt"
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print("Loaded current_best.pt for AI opponent.")
            except:
                print("Could not load current_best.pt.")
        
        self.update_probability()

    def update_probability(self):
        with torch.no_grad():
            tensor = state_to_tensor(self.game).unsqueeze(0)
            _, value = self.model(tensor)
            val = value.item()
            if self.game.current_turn == engine.Player.DEFENDER:
                val = -val
            
            current_move_count = self.replay_index + 1 if self.mode == REPLAY else len(self.game.history_hashes) - 1
            if current_move_count < 0: current_move_count = 0

            if len(self.probabilities) <= current_move_count:
                self.probabilities.append(val)
            else:
                self.probabilities[current_move_count] = val

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.text = "Pause" if self.is_playing else "Play"

    def fast_forward(self):
        if self.playback_speed < 16.0:
            self.playback_speed *= 2.0
        else:
            self.playback_speed = 1.0
        self.btn_ff.text = f"Speed: {self.playback_speed}x"

    def update_replay(self, current_time):
        if self.is_playing:
            interval = 1000 / self.playback_speed
            if current_time - self.last_move_time > interval:
                if self.replay_index < len(self.replay_moves) - 1:
                    self.next_move()
                    self.last_move_time = current_time
                else:
                    self.is_playing = False
                    self.btn_play.text = "Play"

    def draw_board(self, win):
        win.fill(BOARD_COLOR)
        for row in range(ROWS):
            for col in range(COLS):
                rect = (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(win, LINE_COLOR, rect, 1)
                if (row, col) in [(0,0), (0,10), (10,0), (10,10), (5,5)]:
                    pygame.draw.rect(win, (100, 50, 0), rect, 0)

    def draw_pieces(self, win):
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.game.get_piece(row, col)
                if piece != engine.Piece.EMPTY:
                    x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                    y = row * SQUARE_SIZE + SQUARE_SIZE // 2
                    radius = SQUARE_SIZE // 2 - 8
                    
                    if piece == engine.Piece.ATTACKER:
                        color = ATTACKER_COLOR
                    else: # DEFENDER or KING
                        color = KING_SIDE_COLOR
                        
                    pygame.draw.circle(win, color, (x, y), radius)
                    
                    if piece == engine.Piece.KING:
                        line_len = radius - 4
                        pygame.draw.line(win, KING_CROSS_COLOR, (x - line_len, y), (x + line_len, y), 3)
                        pygame.draw.line(win, KING_CROSS_COLOR, (x, y - line_len), (x, y + line_len), 3)

    def draw_highlight(self, win):
        if self.selected_piece:
            r, c = self.selected_piece
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            s.fill(HIGHLIGHT_COLOR)
            win.blit(s, (c * SQUARE_SIZE, r * SQUARE_SIZE))

    def draw_bottom_bar(self, win, mouse_pos):
        pygame.draw.rect(win, (30, 30, 30), (0, BOARD_HEIGHT, WIDTH, BOTTOM_BAR_HEIGHT))
        self.btn_load.draw(win, mouse_pos)
        self.btn_play_ai.draw(win, mouse_pos)
        self.btn_start.draw(win, mouse_pos)
        self.btn_prev.draw(win, mouse_pos)
        self.btn_next.draw(win, mouse_pos)
        self.btn_play.draw(win, mouse_pos)
        self.btn_ff.draw(win, mouse_pos)
        
        # Info text
        turn_text = f"Turn: {self.replay_index + 2 if self.mode == REPLAY else len(self.game.history_hashes)}"
        if self.game.winner != engine.Player.NONE:
            turn_text = f"WINNER: {self.game.winner.name}"
        elif self.ai_is_searching:
            turn_text = "AI is thinking..."
        txt_surf = self.font.render(turn_text, True, TEXT_COLOR)
        win.blit(txt_surf, (10, BOARD_HEIGHT + 210)) 
        
        # Calculate captured pieces
        attacker_count = 0
        defender_count = 0
        for r in range(ROWS):
            for c in range(COLS):
                p = self.game.get_piece(r, c)
                if p == engine.Piece.ATTACKER:
                    attacker_count += 1
                elif p == engine.Piece.DEFENDER:
                    defender_count += 1
        
        attackers_captured = 24 - attacker_count
        defenders_captured = 12 - defender_count
        cap_font = pygame.font.SysFont('Arial', 18)
        capture_text = f"Captured - Attackers: {attackers_captured}  |  Defenders: {defenders_captured}"
        cap_surf = cap_font.render(capture_text, True, (200, 200, 200))
        win.blit(cap_surf, (300, BOARD_HEIGHT + 215))
        
        # Chart
        chart_rect = pygame.Rect(10, BOARD_HEIGHT + 60, WIDTH - 20, 140)
        pygame.draw.rect(win, CHART_BG_COLOR, chart_rect)
        pygame.draw.line(win, (100, 100, 100), (chart_rect.left, chart_rect.centery), (chart_rect.right, chart_rect.centery), 1)
        
        # Chart Labels
        label_font = pygame.font.SysFont('Arial', 16)
        attacker_label = label_font.render("Attackers Win (+1.0)", True, ATTACKER_COLOR)
        defender_label = label_font.render("Defenders Win (-1.0)", True, (200, 200, 200))
        win.blit(attacker_label, (chart_rect.left + 5, chart_rect.top + 5))
        win.blit(defender_label, (chart_rect.left + 5, chart_rect.bottom - 20))

        half = chart_rect.height / 2
        if len(self.probabilities) > 1:
            points = []
            max_scale = max(200, len(self.replay_moves) + 1 if self.mode == REPLAY else len(self.probabilities))
            dx = chart_rect.width / (max_scale - 1)
            for i, p in enumerate(self.probabilities):
                p_clamped = max(-1.0, min(1.0, p))
                y = chart_rect.centery - p_clamped * half
                points.append((chart_rect.left + i * dx, y))
            
            if len(points) > 1:
                pygame.draw.lines(win, CHART_LINE_COLOR, False, points, 2)
            
            curr_idx = self.replay_index + 1 if self.mode == REPLAY else len(self.game.history_hashes) - 1
            if 0 <= curr_idx < len(points):
                pygame.draw.circle(win, (255, 0, 0), (int(points[curr_idx][0]), int(points[curr_idx][1])), 5)

    def draw_side_selection(self, win, mouse_pos):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        win.blit(overlay, (0, 0))
        
        box_rect = pygame.Rect(WIDTH//2 - 200, HEIGHT//2 - 100, 400, 200)
        pygame.draw.rect(win, (50, 50, 50), box_rect)
        pygame.draw.rect(win, (255, 255, 255), box_rect, 2)
        
        title = self.font.render("Select Your Side", True, (255, 255, 255))
        win.blit(title, title.get_rect(center=(WIDTH//2, HEIGHT//2 - 70)))
        
        # Atk Button
        color_atk = (80, 80, 80) if self.selection_rect_atk.collidepoint(mouse_pos) else (40, 40, 40)
        pygame.draw.rect(win, color_atk, self.selection_rect_atk)
        pygame.draw.rect(win, (255, 255, 255), self.selection_rect_atk, 2)
        txt_atk = self.font.render("Attackers", True, (255, 255, 255))
        sub_atk = pygame.font.SysFont('Arial', 14).render("(Black/First)", True, (200, 200, 200))
        win.blit(txt_atk, txt_atk.get_rect(center=(self.selection_rect_atk.centerx, self.selection_rect_atk.centery - 10)))
        win.blit(sub_atk, sub_atk.get_rect(center=(self.selection_rect_atk.centerx, self.selection_rect_atk.centery + 15)))

        # Def Button
        color_def = (80, 80, 80) if self.selection_rect_def.collidepoint(mouse_pos) else (40, 40, 40)
        pygame.draw.rect(win, color_def, self.selection_rect_def)
        pygame.draw.rect(win, (255, 255, 255), self.selection_rect_def, 2)
        txt_def = self.font.render("Defenders", True, (255, 255, 255))
        sub_def = pygame.font.SysFont('Arial', 14).render("(White/Second)", True, (200, 200, 200))
        win.blit(txt_def, txt_def.get_rect(center=(self.selection_rect_def.centerx, self.selection_rect_def.centery - 10)))
        win.blit(sub_def, sub_def.get_rect(center=(self.selection_rect_def.centerx, self.selection_rect_def.centery + 15)))

    def handle_click(self, row, col):
        if self.mode == REPLAY or self.showing_side_selection: return
        
        clicked_piece = self.game.get_piece(row, col)
        
        if clicked_piece != engine.Piece.EMPTY:
            owner = engine.Player.ATTACKER if clicked_piece == engine.Piece.ATTACKER else engine.Player.DEFENDER
            if owner == self.game.current_turn and owner == self.human_side:
                self.selected_piece = (row, col)
        elif self.selected_piece:
            sr, sc = self.selected_piece
            legal_moves = self.game.get_legal_moves()
            target_move = None
            for m in legal_moves:
                if m.from_row == sr and m.from_col == sc and m.to_row == row and m.to_col == col:
                    target_move = m
                    break
            
            if target_move:
                self.game.apply_move(target_move)
                self.selected_piece = None
                self.update_probability()

    def start_ai_game(self, human_side):
        self.mode = ONE_PLAYER_VS_AI
        self.human_side = human_side
        self.game.reset()
        self.probabilities = [0.0]
        self.update_probability()
        self.showing_side_selection = False
        side_name = "Attackers (Black)" if human_side == engine.Player.ATTACKER else "Defenders (White)"
        print(f"Switched to ONE_PLAYER_VS_AI Mode. You are {side_name}.")

    def load_game(self, filename=None):
        if filename is None:
            if not os.path.exists("saves"): return
            files = [os.path.join("saves", f) for f in os.listdir("saves") if f.endswith(".json")]
            if not files: return
            filename = max(files, key=os.path.getctime)

        if os.path.exists(filename):
            with open(filename, "r") as f:
                self.replay_moves = json.load(f)
            self.mode = REPLAY
            self.game.reset()
            self.replay_index = -1
            self.probabilities = []
            self.precalculate_probabilities()
            print(f"Loaded {filename} ({len(self.replay_moves)} moves).")

    def precalculate_probabilities(self):
        temp_game = engine.GameState()
        self.probabilities = []
        def get_val(g):
            with torch.no_grad():
                tensor = state_to_tensor(g).unsqueeze(0)
                _, value = self.model(tensor)
                val = value.item()
                if g.current_turn == engine.Player.DEFENDER: val = -val
                return val
        self.probabilities.append(get_val(temp_game))
        for m_data in self.replay_moves:
            m = engine.Move(m_data["from"][0], m_data["from"][1], m_data["to"][0], m_data["to"][1])
            temp_game.winner = engine.Player.NONE
            temp_game.apply_move(m)
            self.probabilities.append(get_val(temp_game))

    def go_to_start(self):
        if self.mode == REPLAY:
            self.game.reset()
            self.replay_index = -1
            self.is_playing = False
            self.btn_play.text = "Play"

    def prev_move(self):
        if self.mode == REPLAY and self.replay_index >= 0:
            target_idx = self.replay_index - 1
            self.game.reset()
            for i in range(target_idx + 1):
                m_data = self.replay_moves[i]
                m = engine.Move(m_data["from"][0], m_data["from"][1], m_data["to"][0], m_data["to"][1])
                self.game.winner = engine.Player.NONE
                self.game.apply_move(m)
            self.replay_index = target_idx
            self.update_probability()

    def next_move(self):
        if self.mode == REPLAY and self.replay_index < len(self.replay_moves) - 1:
            self.replay_index += 1
            m_data = self.replay_moves[self.replay_index]
            m = engine.Move(m_data["from"][0], m_data["from"][1], m_data["to"][0], m_data["to"][1])
            self.game.winner = engine.Player.NONE
            self.game.apply_move(m)
            self.update_probability()

def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AlphaTafl Visualizer")
    clock = pygame.time.Clock()
    gui = HnefataflGUI()

    run = True
    while run:
        clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: run = False
            if event.type == pygame.DROPFILE: gui.load_game(event.file)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if gui.showing_side_selection:
                    if gui.selection_rect_atk.collidepoint(mouse_pos): gui.start_ai_game(engine.Player.ATTACKER)
                    elif gui.selection_rect_def.collidepoint(mouse_pos): gui.start_ai_game(engine.Player.DEFENDER)
                    else: gui.showing_side_selection = False
                elif mouse_pos[1] < BOARD_HEIGHT:
                    col, row = mouse_pos[0] // SQUARE_SIZE, mouse_pos[1] // SQUARE_SIZE
                    gui.handle_click(row, col)
                else:
                    if gui.btn_load.is_clicked(event, mouse_pos): gui.load_game()
                    elif gui.btn_play_ai.is_clicked(event, mouse_pos): gui.showing_side_selection = True
                    elif gui.btn_start.is_clicked(event, mouse_pos): gui.go_to_start()
                    elif gui.btn_prev.is_clicked(event, mouse_pos): gui.prev_move()
                    elif gui.btn_next.is_clicked(event, mouse_pos): gui.next_move()
                    elif gui.btn_play.is_clicked(event, mouse_pos): gui.toggle_play()
                    elif gui.btn_ff.is_clicked(event, mouse_pos): gui.fast_forward()

        # K6: Async AI search
        if not gui.showing_side_selection and gui.mode == ONE_PLAYER_VS_AI and gui.game.current_turn != gui.human_side and gui.game.winner == engine.Player.NONE:
            if not gui.ai_is_searching and gui.ai_search_result is None:
                # Start AI search in background thread
                gui.ai_is_searching = True
                print("[GUI] AI is thinking...")
                
                def ai_search_task():
                    from src.training.utils import get_legal_moves_mask
                    from src.model.network import get_move_from_index
                    def eval_fn(state_to_eval):
                        with torch.no_grad():
                            tensor = state_to_tensor(state_to_eval).unsqueeze(0)
                            policy, value = gui.model(tensor)
                            policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                            mask = get_legal_moves_mask(state_to_eval).cpu().numpy()
                            policy *= mask
                            if policy.sum() > 0: policy /= policy.sum()
                            return policy.tolist(), value.item()
                    mcts = engine.MCTS(eval_fn, 1.4)
                    probs = np.array(mcts.search(gui.game, 100))
                    action = np.argmax(probs)
                    r1, c1, r2, c2 = get_move_from_index(action)
                    gui.ai_search_result = (r1, c1, r2, c2)
                    gui.ai_is_searching = False
                
                gui.ai_search_thread = threading.Thread(target=ai_search_task, daemon=True)
                gui.ai_search_thread.start()
            
            elif gui.ai_search_result is not None:
                # Apply the completed move
                r1, c1, r2, c2 = gui.ai_search_result
                gui.game.apply_move(engine.Move(r1, c1, r2, c2))
                gui.update_probability()
                gui.ai_search_result = None
                gui.ai_search_thread = None

        gui.update_replay(pygame.time.get_ticks())
        gui.draw_board(win); gui.draw_highlight(win); gui.draw_pieces(win); gui.draw_bottom_bar(win, mouse_pos)
        if gui.showing_side_selection: gui.draw_side_selection(win, mouse_pos)
        pygame.display.update()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
