# SIMPLE IS GENIUS
import warnings # Control diagnostic output
# Suppress the pynvml deprecation warning used by torch.cuda and others
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")

import torch # Deep learning framework for tensor math and GPU acceleration
import torch.nn as nn # Neural network layers and loss functions
import torch.nn.functional as F # High-performance functional activations
import numpy as np # Array manipulation and board-to-matrix conversions
import chess # Pure Python chess library for move validation and logic
import chess.engine # Protocol handler for communicating with Stockfish via UCI
import random, os, math, multiprocessing, shutil, signal # System utilities

# --- GLOBAL ARCHITECTURAL CONSTANTS ---
OBS_SHAPE = (20, 8, 8) # Input tensor: 20 planes (added En-Passant plane)
MOVE_ACTION_DIM = 4096  # Total possible action space (64 from_sq * 64 to_sq)
NUM_RESIDUAL_BLOCKS = 10 # Depth of the residual tower for deep feature extraction
HIDDEN_DIM = 128 # Width of the residual blocks (filters)
POLICY_DIM = 128 # Compression dimension for the policy head
VALUE_DIM = 64 # Compression dimension for the value head

# --- TRAINING CONFIGURATION ---
STOCKFISH_PATH = shutil.which("stockfish") or "/usr/games/stockfish"
STOCKFISH_LIMIT_SECONDS = 0.5  # Safety time cap (backstop only — depth limit is primary)
STOCKFISH_DEPTH = 14
MULTI_PV_COUNT = 10
MARGIN_CENTIPAWNS = 50
MAX_GAME_STEPS = 400
OPENING_BOOK_PGN = "super_gm_book.pgn"
OPENING_BOOK_PLY_LIMIT_RANGE = (8, 12)

# [Optimization] Pre-generate the REMAP array for O(1) symmetry-based data augmentation (horizontal flip)
REMAP = np.array([((i >> 6) ^ 7) << 6 | ((i & 63) ^ 7) for i in range(4096)], dtype=np.int32)

def board_to_tensor(board: chess.Board): 
    """Encodes the 8x8 chessboard and game state into a 20-layer bit-matrix stack.
    TRANSFORMS TO CANONICAL REPRESENTATION: ALWAYS FROM SIDE-TO-MOVE PERSPECTIVE."""
    tensor = np.zeros((20, 8, 8), dtype=np.float16)
    
    # Identify which color is current player
    is_white = board.turn == chess.WHITE
    
    # 1. Encode Piece Placements (12 planes)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Map absolute colors to relative (US vs THEM)
            # Plane 0-5: Current Player's pieces, Plane 6-11: Opponent's pieces
            p_color = piece.color
            rel_color = 0 if p_color == board.turn else 1
            type_idx = piece.piece_type - 1
            
            # Map absolute square to canonical square (Always rank 1 at bottom for US)
            # If Black to move, we flip the board vertically (Rank Flip)
            # This preserves File-A as File-A, mapping e7 to e2 (canonical US perspective)
            r, c = square // 8, square % 8
            if not is_white: r = 7 - r # Rank Flip
            
            tensor[rel_color * 6 + type_idx, r, c] = 1.0
            
    # 2. Encode Static Game State (7 supplementary layers)
    # Layer 12: Turn (redundant in canonical, but kept for compatibility/consistency)
    if is_white: tensor[12, :, :] = 1.0
    
    # Layer 13-14: US Castling Rights (Kingside, Queenside)
    # Layer 15-16: THEM Castling Rights (Kingside, Queenside)
    if board.has_kingside_castling_rights(board.turn): tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(board.turn): tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(not board.turn): tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(not board.turn): tensor[16, :, :] = 1.0
    
    tensor[17, :, :] = min(board.fullmove_number, 100) / 100.0
    tensor[18, :, :] = min(board.halfmove_clock, 50) / 50.0
    
    # Layer 19: En-Passant square (Canonical Perspective)
    if board.ep_square is not None:
        ep_r, ep_c = board.ep_square // 8, board.ep_square % 8
        if not is_white: ep_r = 7 - ep_r # Rank Flip
        tensor[19, ep_r, ep_c] = 1.0
    
    return tensor

def _score_to_cp(score_obj):
    if score_obj.is_mate():
        return 10000 if score_obj.mate() > 0 else -10000
    return float(score_obj.score() or 0)



def assess_move_quality(move, info, board):
    """Categorizes move quality based on centipawn loss from the best engine move.
    0: Good (<= 50cp), 1: Mediocre (<= 150cp), 2: Blunder (> 150cp)."""
    if not info or not move: return 0
    best_score = -100000
    target_score = -100000
    found_target = False
    
    for entry in info:
        if "pv" in entry and len(entry["pv"]) > 0:
            curr_score = _score_to_cp(entry["score"].pov(board.turn))
            if best_score == -100000: best_score = curr_score
            if entry["pv"][0] == move:
                target_score = curr_score
                found_target = True
                break
    
    # If not found in Top 10, it's likely a blunder
    if not found_target: return 2
    
    drop = best_score - target_score
    if drop <= 50: return 0 # [NORMALIZED] Good <= 50cp loss
    if drop <= 150: return 1 # [NORMALIZED] Mediocre <= 150cp loss
    return 2 # Blunder > 150cp


def select_move_temperature(board, info, sampling_temp=1.0, target_temp=0.1):
    """Smarter sampling using separate temperatures for sampling (variety) and target (distillation).
    Strictly preserves Top 10 ranking for policy distillation."""
    if not info: return None, 0, np.zeros(MOVE_ACTION_DIM, dtype=np.float16), False
    
    candidates = []
    for entry in info:
        if "pv" in entry and len(entry["pv"]) > 0:
            move = entry["pv"][0]
            if move and move in board.legal_moves:
                candidates.append((move, _score_to_cp(entry["score"].pov(board.turn))))
    
    if not candidates:
        return None, 0, np.zeros(MOVE_ACTION_DIM, dtype=np.float16), False
    
    # Strictly limit to Top 10 for "The Move Ranking" alignment
    candidates = candidates[:10]
    moves = [c[0] for c in candidates]
    scores = np.array([c[1] for c in candidates], dtype=np.float32)
    max_score = np.max(scores)
    shifted_scores = scores - max_score
    
    # Helper for Boltzmann calculation
    def get_probs(temp):
        t_factor = 50.0 * max(0.01, temp)
        with np.errstate(over='ignore', invalid='ignore'):
            p = np.exp(shifted_scores / t_factor)
        p_sum = np.sum(p)
        if p_sum == 0 or np.isnan(p_sum):
            p = np.zeros_like(p); p[0] = 1.0
        else: p /= p_sum
        return p

    # Distribution 1: Sampling (Used for the actual move played)
    probs_sampling = get_probs(sampling_temp)
    
    # Distribution 2: Target (Used for the policy label/distillation)
    # We keep target_temp low (0.1) so the bot learns the BEST move is significantly better.
    probs_target = get_probs(target_temp)
    
    # Prepare policy vector with Sharp Targets
    policy_vector = np.zeros(MOVE_ACTION_DIM, dtype=np.float16)
    move_indices = [move_to_action_index(mv, board) for mv in moves]
    policy_vector[move_indices] = probs_target.astype(np.float16)
    
    # Sample actual move from High Variety distribution
    move = np.random.choice(moves, p=probs_sampling)
    is_diverse = (move != moves[0])
    
    return move, max_score, policy_vector, is_diverse

def move_to_action_index(move: chess.Move, board: chess.Board):
    """Maps a move to a 4096 index, relative to the current player (canonical)."""
    from_sq, to_sq = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        # Rank Flip for Black's perspective
        from_sq = (7 - (from_sq >> 3)) << 3 | (from_sq & 7)
        to_sq = (7 - (to_sq >> 3)) << 3 | (to_sq & 7)
    return from_sq << 6 | to_sq

def action_index_to_move(action_index: int, board: chess.Board):
    """Maps a 4096 index back to a chess move, relative to current player."""
    from_sq, to_sq = action_index >> 6, action_index & 63
    if board.turn == chess.BLACK:
        # Un-flip rank for Black's perspective
        from_sq = (7 - (from_sq >> 3)) << 3 | (from_sq & 7)
        to_sq = (7 - (to_sq >> 3)) << 3 | (to_sq & 7)
    
    move = chess.Move(from_sq, to_sq)
    # Promotion handling
    piece = board.piece_at(from_sq)
    if piece and piece.piece_type == chess.PAWN:
        if (to_sq >> 3 == 7 and board.turn == chess.WHITE) or (to_sq >> 3 == 0 and board.turn == chess.BLACK):
            move.promotion = chess.QUEEN
    return move

def flip_tensor(tensor: np.ndarray):
    return np.flip(tensor, axis=2).copy()

def flip_policy_vector(policy: np.ndarray):
    return policy[REMAP]

def _is_flip_safe(board: chess.Board):
    """Horizontal flip is only valid when castling and en passant are symmetric.
    This prevents teaching the model that King+Queenside castling is the same as King+Kingside."""
    if board.ep_square is not None: return False # EP is never symmetric in bit-tensors
    
    # Check if castling rights are symmetric for both sides
    wk = board.has_kingside_castling_rights(chess.WHITE)
    wq = board.has_queenside_castling_rights(chess.WHITE)
    bk = board.has_kingside_castling_rights(chess.BLACK)
    bq = board.has_queenside_castling_rights(chess.BLACK)
    return (wk == wq) and (bk == bq)

def load_opening_book(pgn_path):
    import chess.pgn
    trie = {}
    if os.path.exists(pgn_path):
        try:
            with open(pgn_path) as f:
                while (game := chess.pgn.read_game(f)) is not None:
                    node = trie
                    for move in game.mainline_moves():
                        node = node.setdefault(move, {})
        except Exception as e: print(f"Error loading book: {e}")
    return trie

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block: Dynamically recalibrates channel-wise features."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, channels): # Constructor for a residual bottleneck
        super().__init__() # Call parent torch module constructor
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) # Added SE recalibration

    def forward(self, x): # Execution path for a single residual block
        residual = x # Store input to add back as a skip-connection
        out = F.relu(self.bn1(self.conv1(x))) # Conv -> BN -> ReLU
        out = self.bn2(self.conv2(out)) # Conv -> BN
        out = self.se(out) # Apply SE weighing
        out += residual # The 'Residual' step: add original input back (skip connection)
        return F.relu(out) # Final activation before passing to next block

class ChessActorCritic(nn.Module): # The hybrid Actor-Critic neural brain
    """Deep Residual Network with dual-heads for Policy and Value prediction."""
    def __init__(self, input_shape=OBS_SHAPE, action_dim=MOVE_ACTION_DIM, hidden_dim=HIDDEN_DIM, policy_dim=POLICY_DIM, val_dim=VALUE_DIM, res_blocks=NUM_RESIDUAL_BLOCKS):
        super().__init__() # Initialize base module
        # 1. INITIAL FEATURE EXTRACTION
        self.conv_in = nn.Conv2d(input_shape[0], hidden_dim, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(hidden_dim) # Stabilize initial signals
        
        # 2. RESIDUAL TOWER (The deep feature processor)
        self.res_tower = nn.ModuleList([ResBlock(hidden_dim) for _ in range(res_blocks)])
        
        # 3. POLICY HEAD (Decides which moves are strongest)
        pol_channels = max(1, policy_dim // 64)
        self.policy_conv = nn.Conv2d(hidden_dim, pol_channels, kernel_size=1) # Dimensionality reduction
        self.policy_bn = nn.BatchNorm2d(pol_channels) # Scale features
        self.policy_fc = nn.Linear(pol_channels * 8 * 8, action_dim) # Final move distribution
        
        # 4. VALUE HEAD (Estimates the probability of winning)
        self.val_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1) # Project to scalar field
        self.val_bn = nn.BatchNorm2d(1) # Normalize
        self.val_fc1 = nn.Linear(1 * 8 * 8, val_dim) # Compression layer
        self.val_fc2 = nn.Linear(val_dim, 1) # Final tanh output [-1, 1]

        # 5. QUALITY HEAD (Predicts if previous move was good/mediocre/blunder)
        self.qual_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.qual_bn = nn.BatchNorm2d(1)
        self.qual_fc1 = nn.Linear(1 * 8 * 8, val_dim)
        self.qual_fc2 = nn.Linear(val_dim, 3) # [Good, Mediocre, Blunder]

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_tower: x = block(x)
        
        # Policy: Top 10 Ranking
        p = F.relu(self.policy_bn(self.policy_conv(x))).view(x.size(0), -1)
        policy = self.policy_fc(p)
        
        # Value: Evaluation Bar
        v = F.relu(self.val_bn(self.val_conv(x))).view(x.size(0), -1)
        v = F.relu(self.val_fc1(v))
        value = torch.tanh(self.val_fc2(v))
        
        # Quality: How good was the previous move?
        q = F.relu(self.qual_bn(self.qual_conv(x))).view(x.size(0), -1)
        q = F.relu(self.qual_fc1(q))
        quality = self.qual_fc2(q)
        
        return policy, value, quality

def run_game_worker(worker_id, shared_panic_flag, s_arr, p_arr, v_arr, q_arr, ptr, count, total_samples, diversity_samples, lock):
    """Parallel worker that generates chess games using Stockfish distillation."""
    # Reset signal handlers: workers should die silently from SIGKILL, not raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN) # Ignore Ctrl+C in workers; parent handles it
    try: # Attempt to pin this process to a specific CPU core for cache locality
        os.sched_setaffinity(0, {worker_id % multiprocessing.cpu_count()})
    except OSError:
        pass # Fallback if OS does not support affinity pinning
    except Exception:
        pass # Non-critical: process may already be dead

    # Create NumPy views over the shared memory RawArrays for efficient slicing
    states = np.frombuffer(s_arr, dtype=np.float16).reshape(-1, *OBS_SHAPE)
    policies = np.frombuffer(p_arr, dtype=np.float16).reshape(-1, MOVE_ACTION_DIM)
    values = np.frombuffer(v_arr, dtype=np.float16).reshape(-1, 1)
    qualities = np.frombuffer(q_arr, dtype=np.uint8).reshape(-1, 1)
    max_size = states.shape[0] # Total capacity of the circular buffer

    engine = None # Placeholder for the Stockfish engine instance
    try: # Initialize Stockfish via UCI protocol
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        # Limit resources: 1 thread per worker to avoid CPU starvation
        engine.configure({"Threads": 1, "Skill Level": 20, "Hash": 16})
    except Exception as e:
        print(f"Worker {worker_id}: Stockfish init failed ({e})")
        return # Silent exit if Stockfish is missing (Logged in main)

    opening_book = load_opening_book(OPENING_BOOK_PGN)

    try:
        while True:
            board, trajectory, steps = chess.Board(), [], 0
            book_ply_limit = random.randint(*OPENING_BOOK_PLY_LIMIT_RANGE)
            node = opening_book
            prev_quality = 0 # First state has no "previous move"
            
            # [Synchronization] Notify engine of new game to clear hash/internal state
            try:
                engine.send_ucinewgame()
            except: pass

            while not board.is_game_over() and steps < MAX_GAME_STEPS:
                move, score, soft_p, is_div = None, 0.0, None, False
                
                # Try Opening Book
                if node and steps < book_ply_limit and random.random() > 0.10:
                    candidates = [m for m in node.keys() if m in board.legal_moves]
                    if candidates: 
                        move = random.choice(candidates)
                        node = node[move]
                    else: node = None
                else: node = None

                # Engine Analysis
                try:
                    info = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH, time=STOCKFISH_LIMIT_SECONDS), multipv=MULTI_PV_COUNT)
                    if not info: break
                    
                    # High sampling temp (0.8) in opening for variety, Low (0.4) in midgame for consistency
                    s_temp = 0.8 if steps < 20 else 0.4
                    # Target temp (0.05) is VERY sharp to ensure bot learns exactly what is best
                    t_temp = 0.05
                    engine_move, score, soft_p, _ = select_move_temperature(board, info, sampling_temp=s_temp, target_temp=t_temp)
                    
                    if move is None: move = engine_move
                    
                    # Calculate quality of the move we are ABOUT TO play.
                    # This value will be the target for the NEXT board state.
                    current_move_quality = assess_move_quality(move, info, board)
                    
                    is_div = False
                    if random.random() < 0.10 and steps > 4:
                        legal = list(board.legal_moves)
                        if legal:
                            move = random.choice(legal)
                            is_div = True
                            # Re-assess quality if we deviated
                            current_move_quality = assess_move_quality(move, info, board)
                    
                except Exception as e: 
                    fen = board.fen()
                    print(f"Worker {worker_id} engine error: {e}")
                    print(f"Context: Step {steps}, FEN: {fen}")
                    break

                if move is None: break
                if move not in board.legal_moves:
                    print(f"Worker {worker_id} logic error: selected illegal move {move} in {board.fen()}")
                    break
                    
                is_flip_ok = _is_flip_safe(board)
                # Store: (state, move, score, policy, is_div, is_flip, quality_of_PREVIOUS_move)
                trajectory.append((board_to_tensor(board), move, score, soft_p, is_div, is_flip_ok, prev_quality))
                
                board.push(move)
                prev_quality = current_move_quality
                steps += 1

            if len(trajectory) > 0:
                total_new = len(trajectory) * 2
                l_s = np.zeros((total_new, *OBS_SHAPE), dtype=np.float16)
                l_p = np.zeros((total_new, MOVE_ACTION_DIM), dtype=np.float16)
                l_v = np.zeros((total_new, 1), dtype=np.float16)
                l_q = np.zeros((total_new, 1), dtype=np.uint8)
                div_count, idx = 0, 0

                for i, (s, move, score, p_canonical, is_div, is_flip_ok, q_label) in enumerate(trajectory):
                    # Value for the current player at that time
                    v = np.float16(math.tanh(score / 400.0))
                    
                    # Store original canonical sample
                    l_s[idx], l_p[idx], l_v[idx], l_q[idx] = s, p_canonical, v, q_label
                    idx += 1
                    
                    # Store flipped version (Horizontal symmetry augmentation) if safe
                    if is_flip_ok:
                        l_s[idx], l_p[idx], l_v[idx], l_q[idx] = flip_tensor(s), flip_policy_vector(p_canonical), v, q_label
                        idx += 1
                    
                    if is_div: div_count += (2 if is_flip_ok else 1)
                
                # Truncate unused local buffer slots if any (due to flip-skipping)
                l_s, l_p, l_v, l_q = l_s[:idx], l_p[:idx], l_v[:idx], l_q[:idx]
                total_new = idx
                
                with lock:
                    start = ptr.value
                    indices = [(start + j) % max_size for j in range(total_new)]
                    ptr.value = int((start + total_new) % max_size)
                    count.value = int(min(max_size, count.value + total_new))
                    total_samples.value = int(total_samples.value + total_new)
                    diversity_samples.value = int(diversity_samples.value + div_count)
                    states[indices], policies[indices], values[indices], qualities[indices] = l_s, l_p, l_v, l_q
    finally:
        if engine: engine.quit()
