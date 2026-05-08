Gemini
Hnefatafl Rules: Viking Chess Explained
New chat
My stuff
Notebooks

Calculus, 2nd Edition by Gilbert Strang: Abridged Chapters
New notebook
Gems
Chats
PharmD salary
Pinned chat
Airtable Check-in System Creation
Pinned chat
QR Code Webapp Development Clarifications
Pinned chat
Hnefatafl Rules: Viking Chess Explained
Imagine the ultimate video game. Tailored specially to me and my interests. What does it look like? Create a screenshot of the game
Redraw the attached image in the most clumsy, scribbly, and utterly pathetic way possible. Use a white background, and make it look like it was drawn in MS Paint with a mouse.
Startup VC Funding Conversion Ratios Explained
Nirvana's Post-Nevermind Touring Decisions
Cobain and Love's Enabling Addiction
Is Your Salsa Spoiled?
Go Fuseki: Bases vs. Influence
Instagram Story Reshare Issues Explained
Polymarket and WHCD Early End
Go Opening Strategy: Corners, Sides, Center
AI Resignation and Go Game Analysis
YouTube Music Library Access and Recommendations

Settings & help
Conversation with Gemini
What are the rules for hnefatafl

Hnefatafl (often called "Viking Chess") is a fascinating, asymmetrical strategy game. Because the original rules were lost to history and reconstructed from manuscripts and archaeological finds, there are several variations (like the Copenhagen or Fetlar rules). However, the core gameplay remains the same across most modern versions.  
+1

Here is the standard set of rules based on the most widely played 11x11 board version.

The Objective
The game is asymmetrical, meaning the two players have different pieces and different goals:  

The Defenders (White): Their goal is to help the King escape the board by reaching one of the four corner squares.  

The Attackers (Black): Their goal is to surround and capture the King before he can escape.  

The Board and Setup
The Board: Most commonly played on an 11x11 grid.

The Pieces: 1 King, 12 Defenders, and 24 Attackers.  

The Layout: The King starts on the central square, known as the Throne. His 12 Defenders surround him. The 24 Attackers are divided into four groups of six, placed at the center of each of the board's four edges.  

Restricted Squares: The Throne (center) and the four corner squares are hostile to all pieces except the King. Only the King may land on them, though regular pieces can usually pass over the Throne if it is empty.

Movement
All pieces (King, Defenders, and Attackers) move exactly the same way:

Pieces move orthogonally (up, down, left, or right) across the board.

A piece can move any number of empty spaces in a straight line, exactly like a Rook in traditional Chess.  

Pieces cannot jump over other pieces.

The Attackers always make the first move.

Capturing Regular Pieces
Captures are made using a method called custodian capture (or sandwiching).

Sandwiching: You capture an opponent's piece by moving your piece so that the opponent is trapped between two of your pieces on opposite sides (either horizontally or vertically, but never diagonally).

Active Capture: The capture must be an active move. If a player voluntarily moves their own piece between two enemy pieces, it is not captured.  

Hostile Squares: The corners and the empty Throne act as hostile squares. You can capture an enemy piece by sandwiching them between one of your pieces and a restricted square.

Multiple Captures: It is possible to capture multiple pieces in a single turn if your moving piece creates a sandwich in multiple directions simultaneously.

Capturing the King
The King is tougher to capture than a regular piece.  

In the Open: To capture the King, the Attackers must surround him on all four orthogonal sides.  

Against an Edge or Throne: If the King is standing next to the Throne or the edge of the board, the Attackers only need to surround him on the remaining three sides.

Winning the Game: As soon as the King is successfully captured, the Attackers win immediately.

Winning the Game
Defenders Win: The moment the King's piece lands on one of the four corner squares, the Defenders win.

Attackers Win: The moment the King is surrounded and captured, the Attackers win.

Note on Variations: If you buy a physical copy of the game or play online, double-check the specific rulebook included. Some popular variations (like the Copenhagen rules) introduce "Shield Walls" (capturing multiple pieces on an edge) or dictate that the King can be captured exactly like a regular piece.

If I wanted to train an ML algorithm to simulate the game and learn how to play, similar to alpha go or katago, what would the high level approach look like?

Training an AI to master Hnefatafl using the same principles as AlphaGo, AlphaZero, or KataGo involves leveraging Deep Reinforcement Learning (DRL) combined with Monte Carlo Tree Search (MCTS).

Because Hnefatafl is a game of perfect information (no hidden elements), deterministic (no dice rolls), and zero-sum (one player wins, the other loses), it is a perfect candidate for the AlphaZero-style architecture.

Here is the high-level blueprint for how you would build and train this algorithm.

Phase 1: Building the Environment (The Game Engine)
Before the AI can learn, it needs a lightning-fast, digital environment to play in. You would build a game engine that handles:

State Representation (The Input): The AI doesn't "see" the board like we do. You must encode the board state into tensors (multi-dimensional arrays). For an 11x11 board, the input might be an 11×11×N stack of grids, where N represents "channels" (e.g., a channel showing where all attackers are, one for defenders, one for the King, one for restricted squares, and one indicating whose turn it is).

Action Space: A list of all theoretically possible moves on the board. The engine must be able to instantly output a mask of legal moves for any given state.

The Reward Function: Keep it sparse and simple. +1 for a win, −1 for a loss, and 0 for a draw. (Avoid giving points for capturing pieces, as this can teach the AI to farm captures instead of focusing on the actual win condition).

Phase 2: The Neural Network Architecture
You will build a deep neural network, typically a Residual Neural Network (ResNet), which acts as the "intuition" of the AI.

The network takes the board state (the tensors from Phase 1) as input and splits into two distinct "heads" for its output:

The Policy Head: Outputs a probability distribution over the action space. It essentially says, "Based on this board state, here are the moves I think are most promising to explore."

The Value Head: Outputs a single number between −1 and 1. It evaluates the board and says, "Based on this state, here is the probability that the current player will win the game."

Phase 3: Monte Carlo Tree Search (MCTS)
Standard engines (like Stockfish in its early days) use Minimax to calculate every possible future move. Hnefatafl's branching factor is too high for that. Instead, your AI will use MCTS guided by the neural network to look ahead.

When it's the AI's turn to move, it runs hundreds or thousands of simulations:

Selection & Expansion: It traverses the tree of possible future moves, heavily favoring moves that the Policy Head gave high probabilities.

Evaluation: Once it reaches a new, unexplored board position, it doesn't play out the game to the end (which is computationally expensive). Instead, it asks the Value Head to score the position.

Backup: It passes that score back up the tree, updating the win probabilities of the moves that led there.

After running these simulations, the AI confidently plays the move that was visited the most during MCTS.

Phase 4: Self-Play (The Training Loop)
This is where the actual learning happens. The AI starts with zero knowledge, essentially playing random moves.

Data Generation: The AI plays thousands of games against itself using MCTS to decide its moves.

Recording: For every single move in every game, you save three things: the board state, the search probabilities generated by MCTS, and the final outcome of the game (who eventually won).

Training: You feed this massive dataset back into your neural network.

You train the Policy Head to match the MCTS search probabilities (making the network's raw intuition closer to the calculated MCTS results).

You train the Value Head to correctly predict the actual final outcome of the game from that specific state.

Iteration: The newly updated network is now slightly better at the game. You generate a new batch of self-play games with this updated network, and repeat the cycle. Over days or weeks of compute time, the AI bootstrap its own knowledge from beginner to superhuman.

The Hnefatafl Challenge: Asymmetry
The biggest unique hurdle for Hnefatafl compared to Chess or Go is its asymmetry. Because Attackers and Defenders have entirely different goals and start states, the Value Head might struggle initially to accurately predict win probabilities if it gets confused about which side it is evaluating. To solve this, your state representation must explicitly and heavily weight whose turn it is and ensure the network trains equally on playing both the King's side and the Attacker's side.

What programming languages or ML frameworks (like PyTorch or TensorFlow) are you most comfortable with, and have you built any simpler board game environments before?

Would it be better to train separate algorithms for attack vs defense, or one algorithm that plays both? Maybe have each play 1000 games on one side and then switch? Also would it make sense to incorporate playing against a human during the training?

These are fantastic architectural questions. When adapting a generalized algorithm like AlphaZero to an asymmetric game, these are exactly the design choices you have to make.

Here is the breakdown of why the standard approach leans heavily in one direction for all three of your questions.

1. Separate Algorithms vs. One Single Algorithm
You should use one single algorithm that plays both sides. It might seem intuitive to train one "Attacker Bot" and one "Defender Bot," but doing so is highly inefficient because of how neural networks learn.

Shared Representations: The lower layers of a neural network act as feature extractors. They learn basic board geometry: what an empty lane looks like, how pieces block each other, what a cluster of pieces is, and where the board edges are. This spatial logic is identical for both Attackers and Defenders.  

Double the Compute: If you use two separate networks, both have to independently learn these basic geometric facts from scratch, effectively doubling your required computing power and training time.

Instead, you use a single network, but you add a specific input channel (essentially an extra grid in your input tensor) filled entirely with 1s if it is the Attacker's turn, and 0s if it is the Defender's turn. The network will use its shared understanding of geometry, look at that specific channel to remember which side it is playing, and then branch its logic accordingly.

2. The Training Schedule (1000 games and switch?)
You don't need to switch, because the AI plays against itself in real-time.

In the AlphaZero self-play loop, the algorithm isn't playing against a static opponent. It is playing both sides of the board simultaneously.  

During game 1, the AI looks at the board as the Attacker, runs its simulations, and makes a move. Then, the board state updates, and the exact same AI looks at the board from the perspective of the Defender, runs its simulations, and makes the best counter-move.

Because it is constantly playing itself turn-by-turn, it is always perfectly matched against an opponent of its exact same skill level. This constant, simultaneous tug-of-war is what forces the network to improve continuously. If you locked the network to only play one side for 1000 games against a static version of itself, it would likely overfit to exploit specific weaknesses in that static opponent, rather than learning generalized, robust strategies.

3. Incorporating Human Play During Training
It is highly recommended to exclude humans during the training loop.

There are two major reasons to keep humans out of the self-play loop:

The Bottleneck of Speed: Deep Reinforcement Learning requires a staggering amount of data. A successful model will play tens of millions of games to reach a high level. An AI can play thousands of games a minute; a human takes 15 minutes to play one. Injecting a human into the loop would grind the training process to a near halt.  

Human Bias and Skill Ceilings: The biggest breakthrough from DeepMind was the transition from AlphaGo (which trained on human expert games) to AlphaZero (which trained completely from scratch, tabula rasa). Human play is fundamentally flawed and biased. If you train the AI on human data, it learns human mistakes. By forcing it to learn entirely from self-play, it discovers alien, mathematically optimal strategies that humans never conceived of.

The One Exception (Supervised Bootstrapping): If you are extremely limited on computing power and can't afford to let the AI play millions of games to figure out the basics, you can use a database of recorded human Hnefatafl games to do an initial round of Supervised Learning. This teaches the AI basic competency fast. But once that is done, you should sever it from human data and let it improve strictly through self-play.

If you decide to build this, how much computing power (like access to local GPUs or cloud compute) do you have available, as that will ultimately dictate how complex your network can be?

my computer has a 16gb rtx 5080, i9-14900K, 128 gb ram.

That is an absolute powerhouse of a machine. You have more than enough compute to build and train this entirely locally without ever needing to rent cloud servers.

Here is exactly how your specific hardware maps to the AlphaZero-style training pipeline:

16GB RTX 5080 (The Intuition): The Blackwell architecture and its new-generation tensor cores will absolutely chew through the neural network training phases. While 16GB of VRAM can be a bottleneck for running massive Large Language Models (LLMs) today, a ResNet designed to evaluate an 11x11 board game is computationally tiny by comparison. You will easily fit the model, the optimizer states, and massive training batch sizes directly into the VRAM.  
+1

i9-14900K (The Simulator): Monte Carlo Tree Search is notoriously CPU-intensive because it requires traversing complex decision trees millions of times. The 24 cores and massive thread count on the 14900K will allow you to run dozens of self-play game simulations in parallel, generating training data incredibly fast without the CPU bottlenecking your GPU.

128 GB RAM (The Replay Buffer): This is your secret weapon for deep reinforcement learning. To train effectively without "forgetting" older strategies, the AI needs to randomly sample data from a massive "replay buffer" of its most recent 100,000+ moves. With 128GB, you can store a colossal history window entirely in active system memory. This prevents data from swapping to your SSD and keeps the data-feeding loop lightning-fast.

You could likely train an agent from scratch to a highly competitive, superhuman level in just a matter of days or weeks of continuous self-play on this rig alone.

What programming language are you most comfortable using to build the actual game engine—Python (with PyTorch or TensorFlow) or something lower-level like C++ for maximum simulation speed?

whats better

Assuming you are asking whether it is better to use Python (with PyTorch/TensorFlow) or C++ for building your Hnefatafl AI, the answer comes down to a trade-off between execution speed and development time.

Given that you are trying to build an AlphaZero-style architecture, you actually have two entirely different components to build: the Game Engine/MCTS and the Neural Network.

Here is the breakdown of what is best for each part.

1. The Game Engine and MCTS (The Simulator)
Winner: C++

For the actual rules of Hnefatafl and the Monte Carlo Tree Search, C++ is vastly superior.

The Speed Bottleneck: MCTS requires traversing a massive tree of possibilities and generating millions of potential board states. Python is notoriously slow at these types of iterative, CPU-bound tasks.

Exploiting your i9-14900K: C++ handles multithreading incredibly efficiently. You can max out all 24 cores of your CPU to run parallel game simulations. If you write the engine in pure Python, you will likely hit a massive CPU bottleneck, and your RTX 5080 will be sitting idle waiting for Python to finish simulating the games.

2. The Neural Network (The Intuition)
Winner: Python with PyTorch

When it comes to building and training the deep neural network, Python is the undisputed king.  

PyTorch vs. TensorFlow: Right now, PyTorch is the overwhelming favorite in the AI research and Reinforcement Learning communities. It is far more "Pythonic," much easier to debug, and has a massive ecosystem of pre-built AlphaZero implementations you can reference. TensorFlow is still powerful for enterprise production, but PyTorch is better for building custom architectures from scratch.

GPU Acceleration: Python itself isn't doing the heavy lifting here; PyTorch acts as a wrapper that sends the math directly to your RTX 5080's CUDA cores.  

The "Best" Overall Approach: The Hybrid Solution
If you want the absolute best performance—the same setup DeepMind uses—you do both:

You write the ultra-fast Hnefatafl board logic and the MCTS algorithm in C++.

You write the Neural Network and the training loop in Python (using PyTorch).  

You use a library like Pybind11 to connect them. This allows your Python script to "import" your C++ engine as if it were a normal Python module.  

The "Easy" Alternative:
If writing C++ sounds like a nightmare, you can write the entire thing in pure Python. It will just take significantly longer to train. To speed up a pure Python engine, you can use libraries like Numba (which compiles Python code to machine code just-in-time) or use bitboards (representing the board using 64-bit integers instead of arrays) to make the state generation as fast as possible.  

Are you comfortable writing C++ and creating Python bindings, or would you prefer to stick to pure Python to get a working prototype running faster?

fastest possible. can we make a gui so that I can play on the computer? both 2 player (no ai, users take turns) and 1 player (vs AI with version info)

For the fastest possible setup that will eventually plug straight into your PyTorch/C++ AI, Python with Pygame is the undisputed champion. It is the industry standard for ML researchers building custom environments because it is lightweight, renders instantly, and is incredibly easy to hook up to neural networks.

Here is a complete, runnable boilerplate. It gives you an 11x11 interactive graphical board, sets up the starting pieces, handles mouse clicks for moving pieces, and includes the exact architectural "hook" you need to drop your AI versions in later.

1. Install the Dependency
Open your terminal and run:

Bash
pip install pygame
2. The Game GUI Code (hnefatafl_gui.py)
Copy and paste this into a Python file and run it.

Python
import pygame
import sys

# --- CONFIGURATION ---
WIDTH, HEIGHT = 660, 660
ROWS, COLS = 11, 11
SQUARE_SIZE = WIDTH // COLS

# Colors
BOARD_COLOR = (222, 184, 135) # Light wood
LINE_COLOR = (139, 69, 19)    # Dark wood
ATTACKER_COLOR = (40, 40, 40) # Black
DEFENDER_COLOR = (240, 240, 240) # White
KING_COLOR = (255, 215, 0)    # Gold
HIGHLIGHT_COLOR = (100, 200, 100, 150) # Transparent green

# Game Modes
TWO_PLAYER = 0
ONE_PLAYER_VS_AI = 1

# --- BOARD SETUP (0=Empty, 1=Attacker, 2=Defender, 3=King) ---
# This matrix maps perfectly to the input tensor you will need for PyTorch
initial_board = [
    [0,0,0,1,1,1,1,1,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,2,0,0,0,0,1],
    [1,0,0,0,2,2,2,0,0,0,1],
    [1,1,0,2,2,3,2,2,0,1,1],
    [1,0,0,0,2,2,2,0,0,0,1],
    [1,0,0,0,0,2,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,0,0,0]
]

class HnefataflGame:
    def __init__(self, mode=TWO_PLAYER, ai_version="v1.0"):
        self.board = [row[:] for row in initial_board]
        self.turn = 1 # 1 for Attackers (Black), 2 for Defenders (White)
        self.selected_piece = None
        self.mode = mode
        self.ai_version = ai_version
        
    def draw_board(self, win):
        win.fill(BOARD_COLOR)
        # Draw grid lines
        for row in range(ROWS):
            for col in range(COLS):
                rect = (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(win, LINE_COLOR, rect, 1)
                
                # Draw corner escapes and throne
                if (row, col) in [(0,0), (0,10), (10,0), (10,10), (5,5)]:
                    pygame.draw.rect(win, LINE_COLOR, rect, 0) # Fill them in slightly darker
                    
    def draw_pieces(self, win):
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece != 0:
                    x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                    y = row * SQUARE_SIZE + SQUARE_SIZE // 2
                    radius = SQUARE_SIZE // 2 - 8
                    
                    if piece == 1:
                        color = ATTACKER_COLOR
                    elif piece == 2:
                        color = DEFENDER_COLOR
                    elif piece == 3:
                        color = KING_COLOR
                        
                    pygame.draw.circle(win, color, (x, y), radius)

    def draw_highlight(self, win):
        if self.selected_piece:
            r, c = self.selected_piece
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            s.fill(HIGHLIGHT_COLOR)
            win.blit(s, (c * SQUARE_SIZE, r * SQUARE_SIZE))

    def handle_click(self, row, col):
        # 1. Select a piece
        if self.board[row][col] != 0: 
            # Basic turn logic: restrict who can be selected
            clicked_piece = self.board[row][col]
            if (self.turn == 1 and clicked_piece == 1) or (self.turn == 2 and clicked_piece in [2,3]):
                self.selected_piece = (row, col)
                
        # 2. Move a piece
        elif self.selected_piece:
            sr, sc = self.selected_piece
            # TODO: Add your legal move validation and capture logic here
            self.board[row][col] = self.board[sr][sc]
            self.board[sr][sc] = 0
            self.selected_piece = None
            self.turn = 2 if self.turn == 1 else 1 # Swap turns

    def request_ai_move(self):
        """ This is where your neural network and MCTS will step in. """
        print(f"[{self.ai_version}] AI is calculating its move...")
        # ai_move = my_neural_net.get_best_move(self.board)
        # self.board = apply_move(self.board, ai_move)
        # self.turn = 1
        pass

def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Hnefatafl ML Environment")
    clock = pygame.time.Clock()

    # Change to ONE_PLAYER_VS_AI when you are ready to test the bot
    game = HnefataflGame(mode=TWO_PLAYER, ai_version="AlphaTafl_v0.1") 

    run = True
    while run:
        clock.tick(60) # Lock to 60 FPS
        
        # --- AI TURN HANDLING ---
        if game.mode == ONE_PLAYER_VS_AI and game.turn == 2: # Assuming AI plays Defenders
            game.request_ai_move()
            
        # --- EVENT HANDLING ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                
            if event.type == pygame.MOUSEBUTTONDOWN and (game.mode == TWO_PLAYER or game.turn == 1):
                x, y = pygame.mouse.get_pos()
                col, row = x // SQUARE_SIZE, y // SQUARE_SIZE
                game.handle_click(row, col)

        # --- DRAWING ---
        game.draw_board(win)
        game.draw_highlight(win)
        game.draw_pieces(win)
        pygame.display.update()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
How to Build on This:
The Game Engine Logic: You'll notice the handle_click function currently lets pieces teleport anywhere as long as the space is empty. Your next immediate step is to write a fast get_legal_moves(board_state, piece) function to restrict movement to orthogonal lines and implement the "sandwiching" capture logic.

The Tensor Conversion: When you start training your AI, you will write a quick helper function that takes self.board (the simple 2D array) and expands it into the stacked multi-dimensional PyTorch tensor we talked about previously.

The AI Hook: Once your AI is trained, you just uncomment the logic inside request_ai_move() to pass the board state to your PyTorch model, let the CPU run the MCTS, and return the chosen coordinates to update the GUI.

can you write the list of game rules as an md file

Your Markdown file is ready
MD icon
 hnefatafl_rules 
MD

I have generated a clean, formatted Markdown file containing the standard rules for Hnefatafl, complete with objectives, setup, movement, and capture mechanics. Let me know if you'd like to add anything else to it!





Gemini is AI and can make mistakes.

# Hnefatafl (Viking Chess) - Standard Rules

Hnefatafl is an ancient, asymmetrical strategy game of Norse origin. These rules describe the most commonly played modern version on an 11x11 board.

## 1. The Objective
The game features two distinct sides with different goals:
* **The Defenders (White):** Help the King escape the board by reaching one of the four corner squares.
* **The Attackers (Black):** Surround and capture the King before he can escape.

## 2. The Board and Setup
* **Board:** An 11x11 grid.
* **Pieces:** 1 King, 12 Defenders, and 24 Attackers.
* **Layout:** * The King starts on the central square, known as the **Throne**. 
    * The 12 Defenders surround the King. 
    * The 24 Attackers are divided into four groups of six, placed at the center of each of the board's four edges.
* **Restricted Squares:** The Throne (center) and the four corner squares are hostile. Only the King may land on them. Regular pieces may pass *over* the Throne if it is empty, but cannot land on it.

## 3. Movement
All pieces (King, Defenders, and Attackers) move identically:
* Pieces move orthogonally (up, down, left, or right).
* A piece can move any number of empty spaces in a straight line (like a Rook in Chess).
* Pieces cannot jump over other pieces.
* **The Attackers always make the first move.**

## 4. Capturing Regular Pieces
Captures use a method called **custodian capture** (or sandwiching).
* **Sandwiching:** You capture an opponent's piece by moving your piece so that the opponent is trapped between two of your pieces on opposite sides (horizontally or vertically, not diagonally).
* **Active Capture:** The capture must be an active move. If a player voluntarily moves their own piece between two enemy pieces, it is *not* captured.
* **Hostile Squares:** The four corners and the empty Throne act as hostile squares. You can capture an enemy piece by sandwiching them between one of your pieces and a restricted square.
* **Multiple Captures:** You can capture multiple pieces in a single turn if your moving piece creates a sandwich in multiple directions simultaneously.

## 5. Capturing the King
The King is more heavily fortified than regular pieces:
* **In the Open:** To capture the King, the Attackers must surround him on all **four** orthogonal sides.
* **Against an Edge or Throne:** If the King is standing next to the Throne or the edge of the board, the Attackers only need to surround him on the remaining **three** sides.
* **Win Condition:** As soon as the King is successfully captured, the Attackers win immediately.

## 6. Winning the Game
* **Defenders Win:** The moment the King moves onto one of the four corner squares.
* **Attackers Win:** The moment the King is surrounded and captured.

---
*Note: There are many historical and regional variations of Hnefatafl (e.g., Copenhagen rules, Fetlar rules). Always confirm specific variation rules with your opponent or the digital platform before playing.*
hnefatafl_rules.md
Displaying hnefatafl_rules.md.