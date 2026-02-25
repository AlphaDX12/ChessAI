# SIMPLE IS GENIUS
import sys
import torch
import chess
import numpy as np
import chess_engine
import os
import glob

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "Checkpoint"

def load_model():
    """Loads the absolute latest neural checkpoint for competitive play."""
    # Initialize the brain using the master standardized dimensions
    model = chess_engine.ChessActorCritic().to(DEVICE)
    
    # Scan the Checkpoint directory for the most advanced weight file
    all_pths = glob.glob(os.path.join(CHECKPOINT_DIR, 'model_ep_*.pth'))
    # Filter only files (safety catch for directories)
    files_only = [f for f in all_pths if os.path.isfile(f)]
    # Identify the file with the most recent OS-level 'change' timestamp
    latest_file = max(files_only, key=os.path.getctime, default=None) if files_only else None
    
    if latest_file: # If a checkpoint exists
        # Ingest binary weights into the model's structure
        checkpoint = torch.load(latest_file, map_location=DEVICE, weights_only=True)
        # Handle '_orig_mod.' prefix artifacts left by the torch.compile speedup
        sd = {k.replace('_orig_mod.', ''): v for k, v in (checkpoint['model'] if 'model' in checkpoint else checkpoint).items()}
        model.load_state_dict(sd) # Inject learned weights
        return model, latest_file # Return the finalized engine and the source file info
    return model, None # Fallback to a 'tabula rasa' (dumb) model if no weights found

def get_best_move(model, board):
    """Calculates the single strongest move from the neural policy without deep search."""
    model.eval() # Set model to inference mode
    tensor = chess_engine.board_to_tensor(board) # Convert board to 19-layer bit-matrix
    # Prepare tensor for GPU/CPU inference (standardize to float32)
    s = torch.from_numpy(tensor).unsqueeze(0).to(DEVICE, dtype=torch.float32)
    
    with torch.no_grad(): # Disable gradient history to save VRAM and increase speed
        p_logits, v, _ = model(s) # Pass state through the network
        # Convert raw logits to a probability distribution over the 4096-move space
        policy = torch.softmax(p_logits, dim=1).cpu().numpy()[0]
    
    # 1. GENERATE LEGAL MOVES
    legal_moves = list(board.legal_moves)
    move_probs = []
    # 2. MAP LEGAL MOVES TO POLICY PROBABILITIES
    for move in legal_moves:
        idx = chess_engine.move_to_action_index(move, board) # Map UCI move to index (Canonical)
        move_probs.append((move, policy[idx])) # Pair move with its neural score
    
    # 3. SELECT MAX: Choose the move with the absolute highest probability
    best_move = max(move_probs, key=lambda x: x[1])[0]
    return best_move

def uci_loop():
    """Main communication loop for the Universal Chess Interface (UCI) protocol."""
    model, path = load_model() # Ingest the latest brain
    board = chess.Board() # Initialize the virtual game state
    
    while True: # Listen for commands from the Chess GUI (e.g. Arena, CuteChess)
        line = sys.stdin.readline() # Read input from standard input
        if not line: break # Exit on pipe closure
        
        parts = line.strip().split() # Tokenize the command string
        if not parts: continue # Ignore empty lines
            
        cmd = parts[0] # Identify the primary command verb
        
        if cmd == "uci": # Standard UCI handshake
            print("id name MonkeyChessAi Mark2") # Advertise bot name
            print("id author Mathias & Antigravity") # Advertise creators
            print(f"info string loaded checkpoint: {path}") # Reveal weight source
            print("uciok") # Handshake success
        elif cmd == "isready": # Vitality check from host
            print("readyok") # Signal that we are ready to calculate
        elif cmd == "ucinewgame": # Reset signal for a new match
            board = chess.Board() # Flush previous game state
        elif cmd == "position": # Update the internal board state
            if "startpos" in parts: # Default starting position
                board = chess.Board()
                if "moves" in parts: # Follow up with a sequence of moves
                    try:
                        idx = parts.index("moves")
                        moves_slice = parts[idx + 1:]
                        for move_uci in moves_slice: board.push_uci(move_uci)
                    except (ValueError, IndexError): pass
            elif "fen" in parts: # Custom position via Forsyth-Edwards Notation
                try:
                    idx = parts.index("fen")
                    fen_parts = parts[idx + 1 : idx + 7]
                    board = chess.Board(" ".join(fen_parts))
                    if "moves" in parts: # Apply subsequent moves to FEN base
                        m_idx = parts.index("moves")
                        moves_slice = parts[m_idx + 1:]
                        for move_uci in moves_slice: board.push_uci(move_uci)
                except (ValueError, IndexError): pass
        elif cmd == "go": # Trigger move calculation
            best_move = get_best_move(model, board) # Ask the AI for a decision
            print(f"bestmove {best_move.uci()}") # Return move in coordinate notation
        elif cmd == "quit": # Graceful exit signal
            break
        sys.stdout.flush() # Ensure messages are transmitted immediately

if __name__ == "__main__":
    uci_loop()
