# SIMPLE IS GENIUS
import tkinter as tk # Standard Python UI library for legacy interface
from tkinter import messagebox # Toast-style notifications
import chess # Pure Python chess logic
import torch # Neural network weights loading
import uci_engine # Bridge to our optimized model loader
import chess_engine # Board tensorization utilities
import threading # Background process management

class ChessGUI: # Legacy Tkinter-based board for quick testing
    """A minimal, robust GUI for testing the AI outside of the heavy Dashboard."""
    def __init__(self, root): # Constructor for the Tkinter window
        self.root = root 
        self.root.title("Antigravity Chess GPT") # Set title
        
        self.board = chess.Board() # Initialize rule engine
        # Load the latest neural checkpoint from the standard directory
        self.model, _ = uci_engine.load_model()
        self.model.eval() # Set to inference mode
        
        self.selected_square = None # Track piece pick-ups
        self.buttons = [[None for _ in range(8)] for _ in range(8)] # Grid of UI buttons
        
        self.create_widgets() # Build layout
        self.update_board() # Render first frame

    def create_widgets(self):
        self.frame = tk.Frame(self.root)
        self.frame.pack()
        
        for r in range(8):
            for c in range(8):
                btn = tk.Button(self.frame, text="", font=("Arial", 32), width=3, height=1,
                                command=lambda r=r, c=c: self.on_click(r, c))
                btn.grid(row=r, column=c)
                self.buttons[r][c] = btn
        
        self.status = tk.Label(self.root, text="Your turn (White)", font=("Arial", 14))
        self.status.pack()

    def update_board(self):
        unicode_pieces = {
            'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
            None: ''
        }
        
        for r in range(8):
            for c in range(8):
                idx = chess.square(c, 7-r)
                piece = self.board.piece_at(idx)
                symbol = unicode_pieces[piece.symbol()] if piece else ""
                
                # Colors
                bg = "#eeeed2" if (r + c) % 2 == 0 else "#769656"
                if self.selected_square == idx:
                    bg = "#f6f669" # Highlight selected
                
                self.buttons[r][c].config(text=symbol, bg=bg)

    def on_click(self, r, c):
        square = chess.square(c, 7-r)
        
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
        else:
            move = chess.Move(self.selected_square, square)
            # Check for promotion
            if self.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                if (chess.square_rank(square) == 7 and self.board.turn == chess.WHITE) or \
                   (chess.square_rank(square) == 0 and self.board.turn == chess.BLACK):
                    move.promotion = chess.QUEEN
            
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.update_board()
                
                if not self.board.is_game_over():
                    self.status.config(text="Bot is thinking...")
                    self.root.update()
                    # Run bot move in a short delay so user sees their own move first
                    self.root.after(500, self.bot_move)
                else:
                    self.status.config(text=f"Game Over: {self.board.result()}")
            else:
                self.selected_square = None # Reset if illegal
        
        self.update_board()

    def bot_move(self):
        if self.board.is_game_over():
            return
            
        move = uci_engine.get_best_move(self.model, self.board)
        self.board.push(move)
        self.update_board()
        
        if self.board.is_game_over():
            self.status.config(text=f"Game Over: {self.board.result()}")
        else:
            self.status.config(text="Your turn (White)")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChessGUI(root)
    root.mainloop()
