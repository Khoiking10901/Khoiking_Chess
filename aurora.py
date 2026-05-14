"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   A U R O R A   —   Self-Learning Chess Engine                   ║
║   Designed for Google Colab · No GPU required                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

Aurora learns by playing games against herself, remembering which
moves led to wins and avoiding ones that led to losses. All learning
data persists across sessions via JSON files.

Usage (Google Colab):
    !pip install python-chess
    !python aurora.py

Files created automatically:
    memory.json  — learned move weights per board position
    games.json   — full game history with results
"""

import chess
import json
import random
import math
import os
import time
from datetime import datetime


# ──────────────────────────────────────────────
#  CONFIG — tweak these to your liking
# ──────────────────────────────────────────────

MEMORY_FILE     = "memory.json"   # persistent move weights
GAMES_FILE      = "games.json"    # full game records
TRAINING_GAMES  = 80              # games per training session
MINIMAX_DEPTH   = 2               # search depth (1-3 recommended for Colab)
EXPLORE_RATE    = 0.15            # probability of random exploration move
MAX_MOVES_GAME  = 200             # cap to avoid infinite games
WEIGHT_WIN      = +2.0            # reward for moves in a winning game
WEIGHT_LOSS     = -1.5            # penalty for moves in a losing game
WEIGHT_DRAW     = +0.1            # tiny bonus to avoid pure time-waste draws
MEMORY_CAP      = 50_000          # maximum entries before compression kick-in


# ──────────────────────────────────────────────
#  PIECE-SQUARE TABLES  (White's perspective)
#  Encourage good piece placement without ML
# ──────────────────────────────────────────────

PST_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

PST_ROOK = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

PST_KING_MG = [   # middle game — stay safe
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

# Map piece type → PST
PIECE_PST = {
    chess.PAWN:   PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK:   PST_ROOK,
    chess.QUEEN:  PST_QUEEN,
    chess.KING:   PST_KING_MG,
}

# Material values (centipawns)
PIECE_VALUE = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}


# ──────────────────────────────────────────────
#  JSON HELPERS
# ──────────────────────────────────────────────

def load_json(path: str, default) -> object:
    """Load a JSON file; return default value if missing or corrupt."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        print(f"  [warn] Could not read {path}, starting fresh.")
        return default


def save_json(path: str, data: object) -> None:
    """Atomically write data to a JSON file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, path)


# ──────────────────────────────────────────────
#  MEMORY SYSTEM
# ──────────────────────────────────────────────

class Memory:
    """
    Stores move weights per board state.

    memory_dict layout:
        { "FEN_string": { "e2e4": 3.5, "d2d4": -1.0, ... }, ... }

    FEN is stripped of move counters so transpositions share entries.
    """

    def __init__(self):
        raw = load_json(MEMORY_FILE, [])
        # Convert legacy list format → dict format if needed
        if isinstance(raw, list):
            self.data = self._from_list(raw)
        else:
            self.data = raw  # already dict format

    # ── public API ────────────────────────────

    def get_weight(self, fen: str, move_uci: str) -> float:
        """Return stored weight for (position, move). Default 0."""
        return self.data.get(fen, {}).get(move_uci, 0.0)

    def update(self, fen: str, move_uci: str, delta: float) -> None:
        """Add delta to the weight for (position, move)."""
        if fen not in self.data:
            self.data[fen] = {}
        prev = self.data[fen].get(move_uci, 0.0)
        self.data[fen][move_uci] = prev + delta

    def apply_game(self, move_log: list, result: str) -> None:
        """
        Update weights for all (fen, move) pairs in a finished game.
        move_log: list of {"fen": str, "uci": str}
        result:   "1-0" | "0-1" | "1/2-1/2"
        """
        total = len(move_log)
        for i, entry in enumerate(move_log):
            fen  = entry["fen"]
            uci  = entry["uci"]
            side = entry["side"]  # chess.WHITE or chess.BLACK

            # Determine outcome from this side's perspective
            if result == "1/2-1/2":
                delta = WEIGHT_DRAW
            elif (result == "1-0" and side == chess.WHITE) or \
                 (result == "0-1" and side == chess.BLACK):
                delta = WEIGHT_WIN
            else:
                delta = WEIGHT_LOSS

            # Moves made later in game get slightly higher credit
            recency = 0.5 + 0.5 * (i / max(total - 1, 1))
            self.update(fen, uci, delta * recency)

    def save(self) -> None:
        """Compress memory if too large, then persist to disk."""
        if self._size() > MEMORY_CAP:
            self._compress()
        save_json(MEMORY_FILE, self.data)

    def size(self) -> int:
        return self._size()

    # ── internals ─────────────────────────────

    def _size(self) -> int:
        return sum(len(v) for v in self.data.values())

    def _compress(self) -> None:
        """Drop entries with weight very close to 0 to save space."""
        print("  [memory] Compressing memory …")
        before = self._size()
        new_data = {}
        for fen, moves in self.data.items():
            filtered = {m: w for m, w in moves.items() if abs(w) > 0.05}
            if filtered:
                new_data[fen] = filtered
        self.data = new_data
        after = self._size()
        print(f"  [memory] {before} → {after} entries after compression.")

    def _from_list(self, lst: list) -> dict:
        """Convert old list format to dict format."""
        d = {}
        for entry in lst:
            fen  = entry.get("state", "")
            move = entry.get("move", "")
            w    = entry.get("weight", 0.0)
            if fen and move:
                if fen not in d:
                    d[fen] = {}
                d[fen][move] = d[fen].get(move, 0.0) + w
        return d


# ──────────────────────────────────────────────
#  BOARD EVALUATION
# ──────────────────────────────────────────────

def pst_score(square: int, piece_type: int, color: int) -> int:
    """Return piece-square table bonus for a piece."""
    table = PIECE_PST.get(piece_type, [0] * 64)
    # Flip table for black (mirror vertically)
    idx = square if color == chess.WHITE else chess.square_mirror(square)
    return table[idx]


def evaluate(board: chess.Board) -> int:
    """
    Static evaluation from White's perspective (centipawns).
    Positive = White is better; negative = Black is better.
    """
    if board.is_checkmate():
        # Side to move is in checkmate → they lost
        return -100_000 if board.turn == chess.WHITE else +100_000

    score = 0
    for square, piece in board.piece_map().items():
        val   = PIECE_VALUE.get(piece.piece_type, 0)
        bonus = pst_score(square, piece.piece_type, piece.color)
        total = val + bonus
        score += total if piece.color == chess.WHITE else -total

    # Mobility bonus (number of legal moves favours active side)
    if board.turn == chess.WHITE:
        score += len(list(board.legal_moves)) * 3
    else:
        score -= len(list(board.legal_moves)) * 3

    return score


# ──────────────────────────────────────────────
#  MINIMAX WITH ALPHA-BETA PRUNING
# ──────────────────────────────────────────────

def minimax(board: chess.Board, depth: int,
            alpha: int, beta: int, maximising: bool) -> int:
    """
    Standard alpha-beta minimax.
    Returns score from White's perspective.
    """
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    if maximising:
        best = -math.inf
        for move in board.legal_moves:
            board.push(move)
            val = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            best  = max(best, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return best
    else:
        best = math.inf
        for move in board.legal_moves:
            board.push(move)
            val = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            best = min(best, val)
            beta = min(beta, val)
            if beta <= alpha:
                break
        return best


# ──────────────────────────────────────────────
#  MOVE SELECTION
# ──────────────────────────────────────────────

def fen_key(board: chess.Board) -> str:
    """
    Return FEN stripped of half-move/full-move counters.
    This lets transpositions share the same memory entry.
    """
    parts = board.fen().split(" ")
    return " ".join(parts[:4])  # pieces, turn, castling, ep


def select_move(board: chess.Board, memory: Memory,
                move_number: int) -> chess.Move:
    """
    Pick the best move using a blend of:
      1. Pure exploration (random) with probability EXPLORE_RATE
      2. Memory-weighted selection if positions are known
      3. Minimax for top candidate moves otherwise
    """
    legal = list(board.legal_moves)
    if not legal:
        return None

    # ── 1. Opening randomness (first 4 moves each side) ──────────────
    if move_number <= 8 and random.random() < 0.5:
        return random.choice(legal)

    # ── 2. Pure exploration ───────────────────────────────────────────
    if random.random() < EXPLORE_RATE:
        return random.choice(legal)

    # ── 3. Memory-guided weighted selection ──────────────────────────
    fk = fen_key(board)
    weights = []
    for move in legal:
        w = memory.get_weight(fk, move.uci())
        weights.append(w)

    max_w = max(weights)
    # Shift so the worst move = 0, best is higher
    shifted = [w - min(weights) + 0.1 for w in weights]

    # If there's meaningful memory, use it probabilistically
    if max_w > 0.5 or min(weights) < -0.5:
        total = sum(shifted)
        r     = random.random() * total
        cumul = 0.0
        for move, sw in zip(legal, shifted):
            cumul += sw
            if r <= cumul:
                return move
        return legal[-1]

    # ── 4. Minimax fallback ───────────────────────────────────────────
    maximising = (board.turn == chess.WHITE)
    best_move  = None
    best_score = -math.inf if maximising else math.inf

    for move in legal:
        board.push(move)
        score = minimax(board, MINIMAX_DEPTH - 1,
                        -math.inf, math.inf, not maximising)
        board.pop()

        if maximising and score > best_score:
            best_score = score
            best_move  = move
        elif not maximising and score < best_score:
            best_score = score
            best_move  = move

    return best_move or random.choice(legal)


# ──────────────────────────────────────────────
#  GAME RUNNER
# ──────────────────────────────────────────────

def play_game(memory: Memory) -> dict:
    """
    Play one full game of Aurora vs Aurora.
    Returns a game record dict.
    """
    board    = chess.Board()
    move_log = []   # [{"fen": ..., "uci": ..., "san": ..., "side": ...}]
    san_list = []

    move_number = 0
    while not board.is_game_over() and move_number < MAX_MOVES_GAME:
        fk   = fen_key(board)
        side = board.turn
        move = select_move(board, memory, move_number)
        if move is None:
            break

        san = board.san(move)
        move_log.append({"fen": fk, "uci": move.uci(),
                         "san": san, "side": side})
        san_list.append(san)
        board.push(move)
        move_number += 1

    result = board.result()  # "1-0" | "0-1" | "1/2-1/2" | "*"
    if result == "*":
        result = "1/2-1/2"   # treat unfinished as draw

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "moves":     san_list,
        "result":    result,
        "move_log":  move_log,   # full detail for learning
        "total_moves": move_number,
    }


# ──────────────────────────────────────────────
#  TRAINING LOOP
# ──────────────────────────────────────────────

def training_session(n_games: int = TRAINING_GAMES) -> None:
    """
    Run n_games of self-play, update memory, persist everything.
    """
    print("=" * 60)
    print("  A U R O R A  —  Self-Learning Chess Engine")
    print("=" * 60)

    # Load persistent state
    memory   = Memory()
    all_games: list = load_json(GAMES_FILE, [])

    wins = draws = losses = 0

    for g in range(1, n_games + 1):
        t0 = time.time()
        print(f"\n  Game {g}/{n_games}", end=" … ", flush=True)

        game_record = play_game(memory)
        result      = game_record["result"]
        n_moves     = game_record["total_moves"]

        # Tally results (from White's perspective)
        if result == "1-0":
            wins   += 1
        elif result == "0-1":
            losses += 1
        else:
            draws  += 1

        elapsed = time.time() - t0
        print(f"{result}  ({n_moves} moves, {elapsed:.1f}s)")

        # Update memory from this game
        memory.apply_game(game_record["move_log"], result)

        # Store game (without bulky move_log to keep JSON readable)
        storable = {k: v for k, v in game_record.items()
                    if k != "move_log"}
        all_games.append(storable)

        # Persist after every game so progress is never lost
        memory.save()
        save_json(GAMES_FILE, all_games)

    # Final summary
    total_games = len(all_games)
    mem_size    = memory.size()

    print("\n" + "=" * 60)
    print("  Session complete!")
    print(f"  Results this session  →  W:{wins}  D:{draws}  L:{losses}")
    print(f"  Total games on disk   →  {total_games}")
    print(f"  Memory entries        →  {mem_size:,}")
    print("=" * 60)


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    board = chess.Board()
    memory = Memory()

    while True:
        try:
            command = input()

            if command == "uci":
                print("id name Aurora")
                print("id author Khoiking")
                print("uciok")

            elif command == "isready":
                print("readyok")

            elif command.startswith("position startpos"):
                board.reset()

                if "moves" in command:
                    moves = command.split("moves")[1].strip().split()

                    for move in moves:
                        board.push_uci(move)

            elif command.startswith("go"):
                move = select_move(
                    board,
                    memory,
                    board.fullmove_number
                )

                print(f"bestmove {move}")

            elif command == "quit":
                break

        except EOFError:
            break