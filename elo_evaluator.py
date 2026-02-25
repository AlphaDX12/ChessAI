# SIMPLE IS GENIUS
import os, sys, time, json, glob, signal # Standard system utilities
import psutil, chess, chess.engine, torch # Core libraries for system, chess, and neural nets
import numpy as np # Fast numerical math for Elo calculation
import multiprocessing # Parallel process management
from multiprocessing import Pool # Simplified worker pool orchestration
from chess_engine import ChessActorCritic, OBS_SHAPE, MOVE_ACTION_DIM, board_to_tensor, move_to_action_index # Core AI definitions

# --- Global Config & Paths ---
CHECKPOINT_DIR = "Checkpoint" # Folder containing the trained models
ELO_HISTORY_FILE = os.path.join(CHECKPOINT_DIR, "elo_history.json") # Persistent database of Elo over time
ELO_STATE_FILE = os.path.join(CHECKPOINT_DIR, "elo_eval_state.json") # Temporary file to resume crashed evaluations
STOCKFISH_PATH = "/usr/games/stockfish" # Standard location for the C++ engine
TOTAL_GAMES = 5 # Number of games per evaluation session (keep small for fast updates)
CONCURRENT_GAMES = 5 # Number of CPU workers to use simultaneously
ELO_LEVELS = [1350, 1500, 1750, 2000, 2250, 2500, 2750] # Range of Stockfish strengths to test against

_active_pool = None # Global reference for safe cleanup during crashes

def signal_handler(sig, frame):
    """Gracefully shuts down the worker pool if the user hits Ctrl+C."""
    print(f"\n[Terminating] signal {sig} caught. Finalizing workers...")
    if _active_pool:
        _active_pool.terminate() # Kill sub-processes immediately
        _active_pool.join() # Wait for OS to reclaim resources
    sys.exit(1)

# Register handlers for abrupt termination signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class BotPlayer:
    """Lightweight player optimized for high-speed tournament evaluation."""
    def __init__(self, checkpoint_path):
        self.device = torch.device("cpu") # Use CPU for evaluation to save VRAM
        # Reconstruct the brain using the master standardized dimensions
        self.model = ChessActorCritic().to(self.device)
        # Load weights from the binary checkpoint (weights_only=True for security)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        # Architecture compatibility: Strip '_orig_mod.' prefix if model was torch.compiled
        sd = {k.replace('_orig_mod.', ''): v for k, v in (ckpt['model'] if 'model' in ckpt else ckpt).items()}
        self.model.load_state_dict(sd) # Inject learned weights
        self.model.eval() # Disable training behaviors (Dropout/BatchNorm)

    def get_move(self, board):
        """Picks the single strongest move according to the neural network policy."""
        # Translate board to 13-channel bitboard representation
        tensor = board_to_tensor(board)
        s = torch.from_numpy(tensor).unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad(): # Inference-only block (no gradient memory)
            p_logits, _, _ = self.model(s)
            probs = torch.softmax(p_logits, dim=1).cpu().numpy()[0]
        # Sort legal moves by their policy probability and return the maximum
        return max(board.legal_moves, key=lambda m: probs[move_to_action_index(m, board)])

def play_game(args):
    """Worker function: Executes one full game between the bot and Stockfish."""
    game_id, checkpoint_path, target_elo, bot_is_white = args
    bot = BotPlayer(checkpoint_path) # Spawn locally inside the process
    try:
        # Launch external Stockfish engine via the UCI protocol
        sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": target_elo, "Threads": 1})
        board = chess.Board()
        # Game loop: Continues until mate, stalemate, or draw
        while not board.is_game_over() and board.fullmove_number < 150:
            if board.turn == bot_is_white:
                move = bot.get_move(board) # AI Turn
            else:
                # Stockfish Turn: Limit thinking time for speed
                move = sf.play(board, chess.engine.Limit(time=0.01)).move
            board.push(move)
        
        res = board.result() # Standardized result string (e.g. '1-0', '0-1')
        sf.quit() # Free system resources
        # Translate result to outcome for the bot
        if (res == "1-0" and bot_is_white) or (res == "0-1" and not bot_is_white): return "win"
        return "draw" if res == "1/2-1/2" else "loss"
    except Exception: return "error"

def calculate_elo(wins, draws, losses, avg_opponent_elo):
    """Calculates Bayesian Elo estimation with confidence margins."""
    total = wins + draws + losses
    if total == 0: return 0, 0
    # Expected score between 0.0 and 1.0 (win=1, draw=0.5)
    score_pct = max(0.01, min(0.99, (wins + 0.5 * draws) / total))
    # Elo Equation: RatingDiff = 400 * log10(Score / (1-Score))
    rating_diff = 400 * np.log10(score_pct / (1 - score_pct))
    estimate = int(avg_opponent_elo + rating_diff)
    # Calculate Standard Error for the +/- margin
    se = np.sqrt(score_pct * (1 - score_pct) / total)
    margin = int(1.96 * 400 * se / max(0.01, score_pct * (1 - score_pct) * np.log(10)))
    return estimate, min(margin, 500)

def run_evaluation(resume=False):
    """Main orchestrator: Fetches latest model and runs parallel games."""
    all_pths = glob.glob(os.path.join(CHECKPOINT_DIR, 'model_ep_*.pth'))
    latest_file = max(all_pths, key=os.path.getmtime, default=None)
    if not latest_file: return print("No model found.")

    # State tracking for long-running sessions
    state = {"wins": 0, "draws": 0, "losses": 0, "games_done": 0, "ep": 0}
    
    if resume and os.path.exists(ELO_STATE_FILE):
        try:
            with open(ELO_STATE_FILE, 'r') as f:
                state = json.load(f)
            print(f"Resuming evaluation for Ep {state['ep']}... ({state['games_done']}/{TOTAL_GAMES} games done)")
        except Exception as e:
            print(f"Could not load resume state: {e}")
    else:
        # Logic to identify which episode we are testing
        try: state["ep"] = int(latest_file.split('ep_')[-1].split('.')[0])
        except: pass

    def save_state():
        try:
            with open(ELO_STATE_FILE, 'w') as f:
                json.dump(state, f)
        except: pass

    global _active_pool
    pool = Pool(processes=CONCURRENT_GAMES)
    _active_pool = pool
    
    # Prepare batch of parallel game tasks, skipping already completed ones
    tasks = []
    for i in range(TOTAL_GAMES):
        if i < state["games_done"]: continue
        tasks.append((i, latest_file, ELO_LEVELS[i % len(ELO_LEVELS)], i % 2 == 0))
    
    if not tasks:
        print("All games already completed.")
    else:
        print(f"Running {len(tasks)} remaining games for Ep {state['ep']}...")

    try:
        # Launch parallel games and process results as they finish (out-of-order)
        for result in pool.imap_unordered(play_game, tasks):
            if result in ("win", "draw", "loss"):
                state[result + ("s" if result != "loss" else "es")] += 1
                state["games_done"] += 1
                save_state()
                print(f"Progress: {state['games_done']}/{TOTAL_GAMES} | W:{state['wins']} D:{state['draws']} L:{state['losses']}")
    finally:
        pool.terminate(); pool.join(); _active_pool = None

    # Compute final metrics and save to persistent history
    avg_sf = np.mean([ELO_LEVELS[i % len(ELO_LEVELS)] for i in range(TOTAL_GAMES)])
    final_elo, margin = calculate_elo(state["wins"], state["draws"], state["losses"], avg_sf)
    
    # Append the results to the evaluation log
    history = json.load(open(ELO_HISTORY_FILE)) if os.path.exists(ELO_HISTORY_FILE) else []
    history.append({"ep": state["ep"], "elo": final_elo, "margin": margin, "stats": state, "time": time.time()})
    json.dump(history, open(ELO_HISTORY_FILE, 'w'))
    
    # Clean up state file on successful completion
    if os.path.exists(ELO_STATE_FILE):
        os.remove(ELO_STATE_FILE)
        
    print(f"\n[DONE] Estimated Elo: {final_elo} +/- {margin}")

if __name__ == "__main__":
    resume_flag = "--resume" in sys.argv
    run_evaluation(resume=resume_flag) # Run the skill estimation suite
