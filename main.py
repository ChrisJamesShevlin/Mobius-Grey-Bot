#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mobius-Grey (ML) — Stockfish-free UCI engine

- No Stockfish (local or cloud). Explorer allowed for opening stats only.
- Safety rails, simple heuristic eval, 1-ply lookahead with value,
  shallow quiescence on captures/checks, optional tiny MCTS.
- ML: loads model.pkl (win-prob for side-to-move) if present.
- CSV logging with GameId passthrough; schema matches your workflow.

UCI options expected:
  Hash, Threads, Move Overhead,
  ExplorerTopN, ExplorerMinGames, ExplorerMaxPlies,
  ExplorerWeight (INT percent 0..100), UseMCTS, MCTSSims,
  QuiescenceDepth, ModelPath, GameId, LogPath
"""

import os, sys, shlex, json, csv, math, time, datetime, pickle, threading, random
from typing import Optional, List, Tuple, Dict

import requests
import chess

# --------------------- HTTP / Explorer ---------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mobius-Grey-ML/1.0 (+lichess-bot uci)"})
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "4.0"))
EXPLORER_URL = "https://explorer.lichess.ovh/lichess"

def explorer_request(fen: str, topn: int = 20) -> Optional[dict]:
    params = {"fen": fen, "variant": "standard", "moves": max(1, topn), "topGames": 0, "recentGames": 0}
    try:
        r = SESSION.get(EXPLORER_URL, params=params, timeout=HTTP_TIMEOUT)
        if r.ok: return r.json()
    except Exception:
        pass
    return None

# --------------------- Env helpers ---------------------
def env_i(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except Exception: return default

# Compatibility placeholders
DF_HASH           = env_i("ENGINE_HASH", 128)
DF_THREADS        = env_i("ENGINE_THREADS", 1)

# Tunables (defaults if not set via UCI)
DF_EXPL_TOPN      = env_i("EXPLORER_TOP_MOVES", 24)
DF_EXPL_MING      = env_i("EXPLORER_MIN_GAMES", 6)
DF_EXPL_MAX_PLIES = env_i("EXPLORER_MAX_PLIES", 14)
DF_EXPL_WEIGHT_P  = env_i("EXPLORER_WEIGHT_PCT", 30)   # integer percent

DF_USE_MCTS       = (os.getenv("USE_MCTS", "1") != "0")
DF_MCTS_SIMS      = env_i("MCTS_SIMS", 120)
DF_Q_DEPTH        = env_i("QUIESCENCE_DEPTH", 2)

DF_MOVE_OVERHEAD  = env_i("MOVE_OVERHEAD_MS", 200)

MODEL_PATH_DEFAULT= os.getenv("MODEL_PATH", "model.pkl")
LOG_PATH_DEFAULT  = os.getenv("LOG_PATH", "")  # empty => daily auto file

# --------------------- ML model loader ---------------------
def load_model(path: str) -> Optional[object]:
    try:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None

def model_predict_value(model, feats: List[float]) -> float:
    """Return win probability in [0,1] for side-to-move."""
    try:
        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba([feats])[0][-1]); return min(1.0, max(0.0, p))
        if hasattr(model, "decision_function"):
            z = float(model.decision_function([feats])[0]); return 1.0 / (1.0 + math.exp(-z))
        if hasattr(model, "predict"):
            y = float(model.predict([feats])[0])
            return 1.0 / (1.0 + math.exp(-y)) if abs(y) > 1.0 else (y + 1.0) * 0.5
    except Exception:
        pass
    return 0.5

# --------------------- Heuristic eval ---------------------
PIECE_VAL = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
}

# Lightweight piece-square “centrality” table generated once
def _centrality_table() -> List[int]:
    vals = []
    for sq in range(64):
        f = chess.square_file(sq); r = chess.square_rank(sq)
        # Manhattan distance to center squares (d=0 at e4/d4/e5/d5)
        df = min(abs(f-3), abs(f-4))
        dr = min(abs(r-3), abs(r-4))
        d = df + dr
        vals.append(20 - 5*d)  # 20,15,10,5,0,...
    return vals
CENT = _centrality_table()

def phase_factor(board: chess.Board) -> float:
    # 1.0 opening-ish when many pieces; 0.0 endgame-ish
    return min(1.0, max(0.0, (len(board.piece_map()) - 6) / 24.0))

def material_eval(board: chess.Board) -> int:
    s = 0
    for _, pc in board.piece_map().items():
        s += PIECE_VAL[pc.piece_type] * (1 if pc.color == chess.WHITE else -1)
    return s

def pst_eval(board: chess.Board) -> int:
    s = 0
    for sq, pc in board.piece_map().items():
        idx = sq if pc.color == chess.WHITE else chess.square_mirror(sq)
        # modest piece-dependent scaling
        scale = {chess.PAWN:1.0, chess.KNIGHT:1.3, chess.BISHOP:1.2, chess.ROOK:0.8, chess.QUEEN:0.6, chess.KING:0.7}[pc.piece_type]
        v = int(scale * CENT[idx])
        s += v if pc.color == chess.WHITE else -v
    return s

def mobility_eval(board: chess.Board) -> int:
    return len(list(board.legal_moves))

def king_safety_eval(board: chess.Board) -> int:
    def side(color):
        ksq = board.king(color)
        if ksq is None: return 0
        bonus = 0
        r, f = chess.square_rank(ksq), chess.square_file(ksq)
        if (color==chess.WHITE and r==0) or (color==chess.BLACK and r==7):
            if f <= 2 or f >= 5: bonus += 15
        for df in (-1,0,1):
            for dr in (1,2):
                ff, rr = f+df, r+(dr if color==chess.WHITE else -dr)
                if 0<=ff<8 and 0<=rr<8:
                    p = board.piece_at(chess.square(ff, rr))
                    if p and p.piece_type==chess.PAWN and p.color==color: bonus += 6
        return bonus
    return side(chess.WHITE) - side(chess.BLACK)

def features_from_board(board: chess.Board) -> List[float]:
    stm = 1.0 if board.turn == chess.WHITE else -1.0
    mat = material_eval(board)
    pst_s = pst_eval(board)
    mob = mobility_eval(board)
    ks = king_safety_eval(board)
    ph = phase_factor(board)
    counts = [len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))
              for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)]
    return [
        1.0,
        stm * (mat / 1000.0),
        stm * (pst_s / 800.0) * ph,
        (mob / 40.0),
        stm * (ks / 100.0),
        stm * (counts[0] / 8.0),
        stm * (counts[1] / 4.0),
        stm * (counts[2] / 4.0),
        stm * (counts[3] / 4.0),
        stm * (counts[4] / 2.0),
        ph,
    ]

def simple_eval_cp(board: chess.Board) -> int:
    mat = material_eval(board)
    pst_s = pst_eval(board)
    mob = mobility_eval(board)
    ks  = king_safety_eval(board)
    ph  = phase_factor(board)
    return int(0.9*mat + 0.6*pst_s*ph + 0.2*mob + 0.4*ks)

def ml_value(board: chess.Board, model) -> float:
    feats = features_from_board(board)
    if model is not None:
        return model_predict_value(model, feats)
    cp = simple_eval_cp(board)
    return 1.0 / (1.0 + math.exp(-(cp / 140.0)))

# --------------------- Safety rails ---------------------
def move_is_hang(board: chess.Board, mv: chess.Move) -> bool:
    us = board.turn
    if board.is_capture(mv):  # recaptures checked later
        return False
    tmp = board.copy(stack=False)
    try: tmp.push(mv)
    except Exception: return True
    to_sq = mv.to_square
    opp = not us
    if tmp.is_attacked_by(opp, to_sq) and not tmp.is_attacked_by(us, to_sq):
        return True
    return False

def legal_moves_ordered(board: chess.Board) -> List[chess.Move]:
    moves = list(board.legal_moves)
    def key(m):
        k = 0
        if board.is_capture(m): k += 50
        if board.gives_check(m): k += 20
        if chess.square_file(m.to_square) in (3,4) and chess.square_rank(m.to_square) in (3,4): k += 5
        if m.promotion: k += 80
        return -k
    return sorted(moves, key=key)

# --------------------- Quiescence ---------------------
def quiesce_value(board: chess.Board, model, depth: int) -> float:
    base = ml_value(board, model)
    if depth <= 0: return base
    forcing = [m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)]
    if not forcing: return base
    best = 0.0; first = True
    for mv in forcing:
        tmp = board.copy(stack=False)
        try: tmp.push(mv)
        except Exception: continue
        val = 1.0 - quiesce_value(tmp, model, depth - 1)
        if first or val > best:
            best, first = val, False
    return max(base, best)

# --------------------- Tiny MCTS ---------------------
class MCTS:
    def __init__(self, model, explorer_weight: float, q_depth: int):
        self.Q: Dict[Tuple[str,str], float] = {}
        self.N: Dict[Tuple[str,str], int] = {}
        self.Ns: Dict[str, int] = {}
        self.model = model
        self.c_puct = 1.2
        self.explorer_weight = explorer_weight
        self.q_depth = q_depth

    def policy_prior(self, board: chess.Board, moves: List[chess.Move], explorer: Optional[dict]) -> Dict[str, float]:
        pri = {}
        # Explorer prior
        exp: Dict[str, Tuple[float,float]] = {}
        if explorer and "moves" in explorer:
            tot = 0.0
            for m in explorer["moves"]:
                g = float(m.get("gameCount", 0) or 0.0)
                w = float(m.get("white", 0) or 0.0)
                d = float(m.get("draws", 0) or 0.0)
                allp = max(1.0, w + d + (g - (w + d)))
                wr = w / allp
                exp[m.get("uci","")] = (g, wr); tot += g
            if tot > 0:
                for u in list(exp.keys()):
                    g, wr = exp[u]; exp[u] = (g/tot, wr)

        # ML prior via leaf value of child
        vals = {}
        for mv in moves:
            tmp = board.copy(stack=False)
            try: tmp.push(mv)
            except Exception: continue
            v = quiesce_value(tmp, self.model, self.q_depth)
            vals[mv.uci()] = v

        for mv in moves:
            u = mv.uci()
            mlp = vals.get(u, 0.5)
            eg, ewr = exp.get(u, (0.0, 0.5))
            ep = 0.5*eg + 0.5*ewr
            p = (1.0 - self.explorer_weight) * mlp + self.explorer_weight * ep
            pri[u] = max(1e-6, min(1.0, p))
        s = sum(pri.values())
        if s > 0:
            for k in pri:
                pri[k] /= s
        return pri

    def uct_select(self, fen: str, moves: List[chess.Move], pri: Dict[str, float]) -> chess.Move:
        best, best_score = None, -1e18
        for mv in moves:
            u = mv.uci()
            q = self.Q.get((fen, u), 0.5)
            n = self.N.get((fen, u), 0)
            ns = self.Ns.get(fen, 1)
            p = pri.get(u, 1.0/len(moves))
            ucb = q + 1.2 * p * math.sqrt(ns) / (1 + n)
            if ucb > best_score:
                best, best_score = mv, ucb
        return best or random.choice(moves)

    def simulate(self, board: chess.Board, explorer: Optional[dict], depth_cap: int = 8) -> float:
        fen = board.fen()
        moves = legal_moves_ordered(board)
        if not moves or board.is_game_over():
            res = board.result(claim_draw=True)
            if res == "1-0": return 1.0 if board.turn == chess.BLACK else 0.0
            if res == "0-1": return 1.0 if board.turn == chess.WHITE else 0.0
            return 0.5
        pri = self.policy_prior(board, moves, explorer)
        mv = self.uct_select(fen, moves, pri)
        child = board.copy(stack=False)
        child.push(mv)
        if depth_cap <= 1:
            leaf = quiesce_value(child, self.model, self.q_depth)
            val = 1.0 - leaf
        else:
            val = 1.0 - self.simulate(child, None, depth_cap - 1)
        key = (fen, mv.uci())
        self.N[key] = self.N.get(key, 0) + 1
        self.Ns[fen] = self.Ns.get(fen, 0) + 1
        q_old = self.Q.get(key, 0.5); n = self.N[key]
        self.Q[key] = q_old + (val - q_old) / n
        return val

    def best_move(self, board: chess.Board, explorer: Optional[dict], sims: int) -> chess.Move:
        root_moves = legal_moves_ordered(board)
        if not root_moves: raise StopIteration
        fen = board.fen()
        for _ in range(max(1, sims)):
            self.simulate(board, explorer, depth_cap=8)
        best, best_q = root_moves[0], -1.0
        for mv in root_moves:
            q = self.Q.get((fen, mv.uci()), 0.5)
            if q > best_q:
                best, best_q = mv, q
        return best

# --------------------- Engine ---------------------
class MLEngine:
    def __init__(self):
        self.board = chess.Board()
        self.hash = DF_HASH
        self.threads = DF_THREADS
        self.move_overhead = DF_MOVE_OVERHEAD

        # Options
        self.expl_topn = DF_EXPL_TOPN
        self.expl_min_games = DF_EXPL_MING
        self.expl_max_plies = DF_EXPL_MAX_PLIES
        self.expl_weight = DF_EXPL_WEIGHT_P / 100.0  # 0..1

        self.use_mcts = DF_USE_MCTS
        self.mcts_sims = DF_MCTS_SIMS
        self.q_depth = DF_Q_DEPTH

        self.model_path = MODEL_PATH_DEFAULT
        self.model = load_model(self.model_path)

        # Logging
        self.game_id = ""
        self.log_path = LOG_PATH_DEFAULT
        self._ply = 0
        self._log_path_announced = False

        # Metadata no-ops
        self.uci_limit_strength = False
        self.uci_elo = 2850
        self.uci_showwdl = False
        self.uci_analysemode = False
        self.uci_chess960 = False

        print(f"info string model={'loaded' if self.model else 'none'} path={self.model_path}", flush=True)
        print(f"info string explorer_weight={self.expl_weight:.2f} mcts={self.use_mcts} sims={self.mcts_sims} qd={self.q_depth}", flush=True)

        self.lock = threading.Lock()

    # --------- Logging ----------
    def _effective_log_path(self) -> str:
        return self.log_path or f"alf_log_{datetime.date.today().isoformat()}.csv"

    def _log_csv_row(self, row: dict):
        try:
            path = self._effective_log_path()
            need_header = not os.path.exists(path)
            if not self._log_path_announced:
                print(f"info string logging to {path}", flush=True)
                self._log_path_announced = True
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "ts","game_id","ply","fen","turn",
                    "chosen_uci","mode","explorer_used","candidates_json"
                ])
                if need_header: w.writeheader()
                w.writerow(row)
        except Exception as e:
            print(f"info string log_csv_error={e}", flush=True)

    # --------- UCI protocol ----------
    def uci(self):
        print("id name Mobius-Grey ML", flush=True)
        print("id author chris", flush=True)

        print(f"option name Hash type spin default {self.hash} min 16 max 4096", flush=True)
        print(f"option name Threads type spin default {self.threads} min 1 max 32", flush=True)
        print(f"option name Move Overhead type spin default {self.move_overhead} min 0 max 10000", flush=True)
        print("option name Ponder type check default false", flush=True)

        print(f"option name ExplorerTopN type spin default {self.expl_topn} min 5 max 100", flush=True)
        print(f"option name ExplorerMinGames type spin default {self.expl_min_games} min 0 max 1000", flush=True)
        print(f"option name ExplorerMaxPlies type spin default {self.expl_max_plies} min 0 max 40", flush=True)
        print(f"option name ExplorerWeight type spin default {int(self.expl_weight*100)} min 0 max 100", flush=True)
        print(f"option name UseMCTS type check default {'true' if self.use_mcts else 'false'}", flush=True)
        print(f"option name MCTSSims type spin default {self.mcts_sims} min 10 max 2000", flush=True)
        print(f"option name QuiescenceDepth type spin default {self.q_depth} min 0 max 4", flush=True)
        print(f"option name ModelPath type string default {self.model_path}", flush=True)

        print("option name GameId type string default ", flush=True)
        print(f"option name LogPath type string default {self.log_path or 'alf_log_YYYY-MM-DD.csv'}", flush=True)

        print("option name SyzygyPath type string default ", flush=True)
        print("option name UCI_LimitStrength type check default false", flush=True)
        print("option name UCI_Elo type spin default 2850 min 1100 max 3600", flush=True)
        print("option name UCI_ShowWDL type check default false", flush=True)
        print("option name UCI_AnalyseMode type check default false", flush=True)
        print("option name UCI_Chess960 type check default false", flush=True)

        print("uciok", flush=True)

    def isready(self):
        print("readyok", flush=True)

    def ucinewgame(self):
        self.board.reset(); self._ply = 0

    def setoption(self, name: str, value: str):
        key = name.strip().lower().replace(" ", "")
        try:
            if key == "hash":
                self.hash = int(value)
            elif key == "threads":
                self.threads = int(value)
            elif key == "moveoverhead":
                self.move_overhead = int(value)
            elif key == "ponder":
                pass
            elif key == "explorertopn":
                self.expl_topn = max(1, int(value))
            elif key == "explorermingames":
                self.expl_min_games = max(0, int(value))
            elif key == "explorermaxplies":
                self.expl_max_plies = max(0, int(value))
            elif key == "explorerweight":
                self.expl_weight = max(0.0, min(1.0, float(int(value)) / 100.0))
            elif key == "usemcts":
                self.use_mcts = (str(value).lower() == "true")
            elif key == "mctssims":
                self.mcts_sims = max(10, int(value))
            elif key == "quiescencedepth":
                self.q_depth = max(0, int(value))
            elif key == "modelpath":
                self.model_path = value or self.model_path
                self.model = load_model(self.model_path)
                print(f"info string model={'loaded' if self.model else 'none'} path={self.model_path}", flush=True)
            elif key == "gameid":
                self.game_id = value or ""
            elif key == "logpath":
                self.log_path = value or ""
                self._log_path_announced = False
            elif key == "syzygypath":
                pass
            elif key == "uci_limitstrength":
                self.uci_limit_strength = False
            elif key == "uci_elo":
                self.uci_elo = 2850
            elif key == "uci_showwdl":
                self.uci_showwdl = (str(value).lower() == "true")
            elif key == "uci_analysemode":
                self.uci_analysemode = False
            elif key == "uci_chess960":
                self.uci_chess960 = False
        except Exception:
            pass

    def position(self, tokens: List[str]):
        if not tokens: return
        if tokens[0] == "startpos":
            self.board.reset(); self._ply = 0
            if len(tokens) >= 3 and tokens[1] == "moves":
                for u in tokens[2:]:
                    try: self.board.push_uci(u); self._ply += 1
                    except Exception: break
        elif tokens[0] == "fen":
            fen = " ".join(tokens[1:7])
            try: self.board.set_fen(fen); self._ply = 0
            except Exception: return
            tail = tokens[7:]
            if tail and tail[0] == "moves":
                for u in tail[1:]:
                    try: self.board.push_uci(u); self._ply += 1
                    except Exception: break

    def _explorer_ok(self) -> bool:
        return self._ply < self.expl_max_plies and self.expl_weight > 0.0

    def pick_move(self, movetime_ms: Optional[int], wtime: Optional[int], btime: Optional[int],
                  winc: int, binc: int) -> Optional[chess.Move]:

        under_time = False
        if wtime is not None and btime is not None:
            side_ms = wtime if self.board.turn == chess.WHITE else btime
            if side_ms < (2000 + self.move_overhead): under_time = True

        explorer = None
        if self._explorer_ok() and not under_time:
            explorer = explorer_request(self.board.fen(), self.expl_topn)
            if explorer and "moves" in explorer:
                explorer["moves"] = [m for m in explorer["moves"]
                                     if int(m.get("gameCount",0) or 0) >= self.expl_min_games]
                if not explorer["moves"]: explorer = None

        moves = legal_moves_ordered(self.board)
        safe_moves = [m for m in moves if not move_is_hang(self.board, m)] or moves
        if not safe_moves: return None

        if self.use_mcts and not under_time:
            mcts = MCTS(self.model, explorer_weight=(self.expl_weight if explorer else 0.0), q_depth=self.q_depth)
            chosen = mcts.best_move(self.board, explorer, sims=self.mcts_sims)
            mode = "mcts"
        else:
            best, best_val = None, -1.0
            for mv in safe_moves:
                tmp = self.board.copy(stack=False)
                try: tmp.push(mv)
                except Exception: continue
                v = quiesce_value(tmp, self.model, self.q_depth if not under_time else 0)
                val = 1.0 - v
                if explorer:
                    for m in explorer.get("moves", []):
                        if m.get("uci") == mv.uci():
                            g = float(m.get("gameCount",0) or 0.0)
                            w = float(m.get("white",0) or 0.0); d=float(m.get("draws",0) or 0.0)
                            allp = max(1.0, w + d + (g - (w + d)))
                            wr = w / allp
                            pop = min(1.0, g / 1000.0)
                            prior = 0.5*pop + 0.5*wr
                            val = (1.0 - self.expl_weight) * val + self.expl_weight * prior
                            break
                if val > best_val: best, best_val = mv, val
            chosen = best or safe_moves[0]
            mode = "1ply"

        # Log
        try:
            cand_rows = [{"uci": mv.uci()} for mv in safe_moves[:20]]
            row = {
                "ts": time.time(),
                "game_id": self.game_id,
                "ply": self._ply + 1,
                "fen": self.board.fen(),
                "turn": "w" if self.board.turn else "b",
                "chosen_uci": chosen.uci() if chosen else None,
                "mode": mode,
                "explorer_used": 1 if explorer else 0,
                "candidates_json": json.dumps(cand_rows, separators=(",",":")),
            }
            self._log_csv_row(row)
        except Exception as e:
            print(f"info string log_pack_error={e}", flush=True)

        return chosen

    def go_and_print(self, cmd: str):
        parts = cmd.split()
        def gi(tag): return int(parts[parts.index(tag)+1]) if tag in parts else None
        movetime = gi("movetime"); wtime = gi("wtime"); btime = gi("btime")
        winc = gi("winc") or 0; binc = gi("binc") or 0
        mv = self.pick_move(movetime, wtime, btime, winc, binc)
        print("bestmove " + (mv.uci() if mv else "0000"), flush=True)

    def loop(self):
        while True:
            line = sys.stdin.readline()
            if not line: break
            line = line.strip()
            if not line: continue
            if line == "uci": self.uci()
            elif line == "isready": self.isready()
            elif line == "ucinewgame": self.ucinewgame()
            elif line.startswith("setoption"):
                try: parts = shlex.split(line)
                except Exception: parts = line.split()
                name, value, seen_value = [], None, False
                for tok in parts[1:]:
                    if tok == "name": continue
                    if tok == "value": seen_value = True; continue
                    if seen_value: value = tok if value is None else value + " " + tok
                    else: name.append(tok)
                self.setoption(" ".join(name), value if value is not None else "")
            elif line.startswith("position"): self.position(line.split()[1:])
            elif line.startswith("go"): self.go_and_print(line)
            elif line == "stop": pass
            elif line == "quit": break

# --------------------- entry ---------------------
if __name__ == "__main__":
    MLEngine().loop()
