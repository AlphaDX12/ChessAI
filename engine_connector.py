# SIMPLE IS GENIUS
import torch # Deep learning framework for neural network execution
import os # System-level operations for file paths and processes
import glob # Pattern matching for finding the latest checkpoints
import chess # Pure Python chess library for rule validation
import numpy as np # High-performance vector and array math
import math # Mathematical functions for MCTS
from chess_engine import ChessActorCritic, OBS_SHAPE, MOVE_ACTION_DIM, NUM_RESIDUAL_BLOCKS, board_to_tensor, move_to_action_index # Core AI definitions


class ModelInference: # Central manager for the AlphaZero neural brain
    """The 'Brain' of the application: Connects neural weights to game logic."""
    def __init__(self, device="cpu"): # Constructor for the AI brain
        self.device = torch.device(device) # Determine target hardware (e.g. 'cuda' or 'cpu')
        self.model = None # Handle to the active neural network instance
        self.book = {} # Memory-mapped dictionary for instant opening book lookups

    def load_latest(self, checkpoint_dir="Checkpoint"): # Checkpoint recovery logic
        """Scans the file system for the most advanced trained model available."""
        # Multi-track search for the best candidate
        state_path = os.path.join(checkpoint_dir, "training_state.pth")
        full_ckpts = sorted(glob.glob(f"{checkpoint_dir}/full_state_ep_*.pth"), key=os.path.getmtime)
        model_ckpts = sorted(glob.glob(f"{checkpoint_dir}/model_ep_*.pth"), key=os.path.getmtime)
        
        candidates = []
        if os.path.exists(state_path): candidates.append(state_path)
        candidates += full_ckpts[::-1] # Newest first
        candidates += model_ckpts[::-1]
        
        if not candidates: return False # No brain found

        for path in candidates:
            try: # Attempt to ingest the neural weights
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                self.model = ChessActorCritic().to(self.device)
                sd = checkpoint['model'] if 'model' in checkpoint else checkpoint
                # Strip '_orig_mod.' prefix caused by torch.compile
                new_state = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
                # Use strict=False to allow loading older models that lack the 'quality' head
                self.model.load_state_dict(new_state, strict=False) 
                self.model.eval()
                print(f"Engine Loaded successfully from {os.path.basename(path)}")
                return True 
            except Exception as e:
                print(f"Skipping checkpoint {path}: {e}")
                continue
        return False

    def get_eval(self, board): # Advantage estimation for the UI meter
        """Standardizes neural evaluation for the GUI's linear tension meter."""
        if not self.model: return 0.0 # Return absolute draw if no engine is active
        with torch.no_grad(): # Disable gradient tracking to save VRAM and increase speed
            # Pass board through the Value head to get an outcome prediction (-1.0 to 1.0)
            _, value, _ = self.model(torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0).to(self.device))
        # Neural output is tanh (-1.0 to 1.0). Upscale by 400.0 for familiar Centipawn visualization.
        return float(value.item()) * 400.0 # Return scaled score

    def get_ai_move(self, board, num_sims=100): # Strategic move calculation with MCTS
        """Calculates the best move using Monte Carlo Tree Search for lookahead."""
        # [ROBUSTNESS] Auto-load brain if it wasn't available at startup
        if not self.model:
            self.load_latest()
            
        if not self.model: return None, 0.0, []

        # Use MCTS for search if available, otherwise fallback to raw policy
        mcts = MCTS(self.model, self.device)
        best_move, value_score, top_10 = mcts.search(board, num_sims=num_sims)
        
        return best_move, value_score, top_10

    def get_move_quality(self, board): 
        """Uses the 'Quality' head to assess the move that led to the CURRENT board state.
        Returns the softmax probabilities for [Good, Mediocre, Blunder]."""
        if not self.model: return [1.0, 0.0, 0.0] # Default to 100% Good

        tensor = board_to_tensor(board)
        s = torch.from_numpy(tensor).unsqueeze(0).to(self.device, dtype=torch.float32)
        
        with torch.no_grad():
            _, _, quality = self.model(s)
            probs = torch.softmax(quality, dim=1).cpu().numpy()[0]
        
        return probs.tolist() # [p_good, p_mediocre, p_blunder]

    def get_stockfish_move(self, board, move_time_ms=500): # Integration with classical Stockfish
        """Interprets Stockfish output for competitive benchmarking."""
        import chess.engine
        import shutil
        # Dynamically locate the Stockfish binary
        sf_path = shutil.which("stockfish") or "/usr/games/stockfish"
        
        if not os.path.exists(sf_path):
            print(f"Stockfish Connector Error: Binary not found at {sf_path}")
            return None
            
        try:
            # We use a context manager for one-off evaluation per call (Simple is Genius)
            # This avoids managing a persistent process that could leak or hang the GUI
            with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
                # Limit resources to 1 thread for GUI responsiveness
                engine.configure({"Threads": 1, "Hash": 16})
                # Execute search with the requested time budget
                result = engine.play(board, chess.engine.Limit(time=move_time_ms / 1000.0))
                return result.move
        except Exception as e:
            print(f"Stockfish Connector Error: {e}")
        return None

class MCTSNode:
    """A single node in the MCTS tree, representing a specific board state."""
    def __init__(self, p=0):
        self.p = p # Prior probability from the Policy head
        self.n = 0 # Visit count
        self.w = 0 # Accumulated value (win/loss)
        self.q = 0 # Mean value (w / n)
        self.children = {} # Map of move -> MCTSNode

class MCTS:
    """Monte Carlo Tree Search implementation for AlphaZero-style lookahead."""
    def __init__(self, model, device, c_puct=1.5):
        self.model = model
        self.device = device
        self.c_puct = c_puct

    def search(self, board, num_sims=100):
        root = MCTSNode()
        
        # Initial expansion of the root
        self._expand_node(root, board)
        
        # Run simulations
        for _ in range(num_sims):
            node = root
            search_board = board.copy()
            path = [node]
            
            # 1. Select: Traverse the tree using PUCT until a leaf is found
            while node.children:
                move, node = self._select_child(node)
                search_board.push(move)
                path.append(node)
            
            # 2. Expand & Evaluate: Use NN to get value and policy for the leaf
            if not search_board.is_game_over():
                v = self._expand_node(node, search_board)
            else:
                # Terminal state value
                res = search_board.result()
                if res == "1-0": v = 1.0 if board.turn == chess.WHITE else -1.0
                elif res == "0-1": v = -1.0 if board.turn == chess.WHITE else 1.0
                else: v = 0.0
                # Flip value if it's the opponent's turn in the terminal state
                if search_board.turn != board.turn: v = -v
            
            # 3. Backup: Propagate the value up the tree
            for node in reversed(path):
                node.n += 1
                node.w += v
                node.q = node.w / node.n
                v = -v # Value flips at each level (Minimax perspective)

        # Final move selection: Pick move with highest visit count
        if not root.children:
            return None, 0.0, []
            
        best_move = max(root.children.items(), key=lambda x: x[1].n)[0]
        root_value = root.q * 400.0
        
        # Prepare top 10 list for the UI scoreboard
        # Sort based on visit count (most reliable metric)
        sorted_moves = sorted(root.children.items(), key=lambda x: x[1].n, reverse=True)
        top_10 = []
        for move, node in sorted_moves[:10]:
            p = node.n / root.n # Policy as visit probability
            top_10.append((move, p, node.q * 400.0))
            
        return best_move, root_value, top_10

    def _select_child(self, node):
        """Selected the move that maximizes PUCT (Priority Upper Confidence Bound for Trees)."""
        best_puct = -float('inf')
        best_move = None
        best_child = None
        
        sqrt_n = math.sqrt(node.n)
        for move, child in node.children.items():
            # PUCT Formula: Q + C_puct * P * (sqrt(N_parent) / (1 + n_child))
            puct = child.q + self.c_puct * child.p * (sqrt_n / (1 + child.n))
            if puct > best_puct:
                best_puct = puct
                best_move = move
                best_child = child
        return best_move, best_child

    def _expand_node(self, node, board):
        """Uses the Neural Network to populate leaf node's children and priors."""
        tensor = board_to_tensor(board)
        s = torch.from_numpy(tensor).unsqueeze(0).to(self.device, dtype=torch.float32)
        
        with torch.no_grad():
            p_logits, v, _ = self.model(s)
            probs = torch.softmax(p_logits, dim=1).cpu().numpy()[0]
            val = v.item()

        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            p = probs[move_to_action_index(move, board)]
            node.children[move] = MCTSNode(p)
            
        return val
