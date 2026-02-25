
import numpy as np
import chess
from chess_engine import select_move_temperature

def test_extreme_scores():
    print("Testing select_move_temperature with extreme scores...")
    
    # Mock 'info' object as returned by engine
    class MockScore:
        def __init__(self, cp): self.cp = cp
        def pov(self, color): return self
        def score(self): return self.cp
        def is_mate(self): return False

    class MockEntry:
        def __init__(self, move, cp):
            self.move = move
            self.score_obj = MockScore(cp)
        
        def __getitem__(self, key):
            if key == "pv": return [self.move]
            if key == "score": return self.score_obj
            return None

        def __contains__(self, key):
            return key in ["pv", "score"]

    board = chess.Board()
    
    # Test 1: Normal scores
    info = [MockEntry(chess.Move.from_uci("e2e4"), 50), MockEntry(chess.Move.from_uci("d2d4"), 40)]
    move, best, policy, diverse = select_move_temperature(board, info, temp=1.0)
    print(f"Test 1 (Normal): best={best}, probs={policy[policy > 0]}")
    assert not np.isnan(policy).any(), "NaN found in normal test"

    # Test 2: Extreme winning score (Mate-like)
    info = [MockEntry(chess.Move.from_uci("e2e4"), 10000), MockEntry(chess.Move.from_uci("d2d4"), -10000)]
    move, best, policy, diverse = select_move_temperature(board, info, temp=0.1)
    print(f"Test 2 (Extreme Win): best={best}, probs={policy[policy > 0]}")
    assert not np.isnan(policy).any(), "NaN found in extreme win test"
    assert policy.sum() > 0.99, "Probabilities don't sum to 1"

    # Test 3: Extreme losing score (Mate-like)
    info = [MockEntry(chess.Move.from_uci("e2e4"), -10000), MockEntry(chess.Move.from_uci("d2d4"), -10050)]
    move, best, policy, diverse = select_move_temperature(board, info, temp=0.1)
    print(f"Test 3 (Extreme Loss): best={best}, probs={policy[policy > 0]}")
    assert not np.isnan(policy).any(), "NaN found in extreme loss test"

    # Test 4: All equal scores
    info = [MockEntry(chess.Move.from_uci("e2e4"), 0), MockEntry(chess.Move.from_uci("d2d4"), 0)]
    move, best, policy, diverse = select_move_temperature(board, info, temp=1.0)
    print(f"Test 4 (Equal): best={best}, probs={policy[policy > 0]}")
    assert np.allclose(policy[policy > 0], 0.5, atol=1e-2)

    print("All numerical stability tests passed!")

if __name__ == "__main__":
    test_extreme_scores()
