#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mobius-Grey (ALF) v3 — Explorer + Cloud Eval + Stockfish blended UCI engine
with Phase-1 ML data capture (CSV logger) and optional tiny ML reranker

- Answers UCI (uci/isready/ucinewgame/position/go/stop/quit)
- Uses Stockfish MultiPV as a base
- Adds Explorer (book) EV and Cloud Eval bonuses to candidate moves
- Clean fallback to pure Stockfish
- Accepts lichess-bot's common options (Hash, Threads, Move Overhead, Ponder)
- Extra tunables as UCI options (ExplorerWeight, CloudWeight, MinGames, SFDepth, SFMultiPV, CloudMultiPV, ExplorerTopN)
- Reranker: UseReranker (tiny ML tie-window reranker), TieWindowCP (window around SF best)
- v3 NEW: Phase-1 ML logging to CSV (auto-created daily or at LogPath), GameId passthrough
- v3 NEW: Prints info about reranker mode and log path
- v2 goodies retained: time-pressure skipping of online calls, LRU cache for Explorer/Cloud, SyzygyPath passthrough

Requires:
    pip install python-chess requests
    sudo apt install stockfish   (or set env STOCKFISH_PATH)

Optional:
    - Drop a scikit/LightGBM-style pickle at ./alf_rerank.pkl or set env ALF_RERANK_MODEL
      to enable learned reranking. Otherwise a tiny linear fallback is used.
"""

import os
import sys
import shlex
import shutil
import threading
import math
import pickle
import csv
import json
import time
import datetime
from functools import lru_cache
from typing import Optional, Dict, List, Tuple

import requests
import chess
import chess.engine

# --------------------- HTTP setup ---------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ALF-ExplorerCloud/1.0 (+lichess-bot uci)"})
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "5.0"))

EXPLORER_URL = "https://explorer.lichess.ovh/lichess"
CLOUD_URL    = "https://lichess.org/api/cloud-eval"

# --------------------- defaults / env -----------------
def env_i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def env_f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

# Stockfish / search
DF_HASH      = env_i("SF_HASH", 128)
DF_THREADS   = env_i("SF_THREADS", 1)
DF_SF_DEPTH  = env_i("SF_DEPTH", 16)
DF_SF_MPVS   = env_i("SF_MULTIPV", 8)

# Explorer / cloud
DF_EXPL_TOPN = env_i("EXPLORER_TOP_MOVES", 30)
DF_EXPL_MING = env_i("EXPLORER_MIN_GAMES", 10)
DF_EXPL_W    = env_f("EXPLORER_WEIGHT_CP", 50.0)   # abs cap of cp bonus
DF_CLOUD_MPVS= env_i("CLOUD_MULTIPV", 6)
DF_CLOUD_W   = env_f("CLOUD_WEIGHT_CP", 20.0)      # cp/5 capped

# --------------------- helpers ------------------------
def find_stockfish() -> Optional[str]:
    envp = os.getenv("STOCKFISH_PATH")
    if envp and os.path.isfile(envp) and os.access(envp, os.X_OK):
        return envp
    p = shutil.which("stockfish")
    if p: return p
    p = "/usr/games/stockfish"
    if os.path.isfile(p) and os.access(p, os.X_OK): return p
    return None

def cp_from_score(score: chess.engine.PovScore, white_to_move: bool) -> int:
    pov = chess.WHITE if white_to_move else chess.BLACK
    s = score.pov(pov)
    if s.is_mate():
        m = s.mate()
        return 100000 if (m and m > 0) else -100000
    val = s.score(mate_score=100000)
    return int(val or 0)

def _safe_load_model():
    """Optionally load a pickled model from env ALF_RERANK_MODEL or ./alf_rerank.pkl."""
    path = os.getenv("ALF_RERANK_MODEL", "alf_rerank.pkl")
    try:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

# --------------------- online data (+ LRU cache) --------------------
def explorer_request(fen: str, topn: int) -> Optional[dict]:
    params = {"fen": fen, "variant": "standard", "moves": max(1, topn), "topGames": 0, "recentGames": 0}
    try:
        r = SESSION.get(EXPLORER_URL, params=params, timeout=HTTP_TIMEOUT)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None

def cloud_request(fen: str, multi_pv: int) -> Optional[dict]:
    params = {"fen": fen, "multiPv": max(1, multi_pv)}
    try:
        r = SESSION.get(CLOUD_URL, params=params, timeout=HTTP_TIMEOUT)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None

@lru_cache(maxsize=8192)
def explorer_request_cached(fen: str, topn: int) -> dict:
    return explorer_request(fen, topn) or {}

@lru_cache(maxsize=8192)
def cloud_request_cached(fen: str, multi_pv: int) -> dict:
    return cloud_request(fen, multi_pv) or {}

def explorer_bonus_for(uci: str, ex: dict, min_games: int, cap_cp: float) -> float:
    """Return cp-like EV from Explorer for a specific move (0 if below min_games)."""
    if not ex or "moves" not in ex:
        return 0.0
    for m in ex["moves"]:
        if m.get("uci") == uci:
            g = int(m.get("gameCount", 0))
            w = float(m.get("white", 0)); d = float(m.get("draws", 0)); b = float(m.get("black", 0))
            tot = max(1.0, w + d + b)
            winP  = 100.0 * w / tot
            drawP = 100.0 * d / tot
            # cp-ish: prefer wins, slight penalty for drawiness
            ev = (winP - 50.0) - 0.35 * (drawP - 30.0)
            ev = max(-cap_cp, min(cap_cp, ev))
            return ev if g >= min_games else 0.0
    return 0.0

def cloud_bonus_for(uci: str, cloud: dict, cap_cp: float) -> float:
    """Small bonus based on Cloud Eval PV cp -> cp/5, capped."""
    if not cloud or "pvs" not in cloud:
        return 0.0
    best = 0.0
    for pv in cloud.get("pvs", []):
        moves = pv.get("moves", "")
        if not moves: continue
        first = moves.split()[0]
        if first == uci:
            cp = float(pv.get("cp", 0.0))
            bonus = max(-cap_cp, min(cap_cp, cp / 5.0))
            best = max(best, bonus)
    return best

# --------------------- Engine -------------------------
class ALFEngine:
    def __init__(self):
        self.board = chess.Board()
        # UCI options (must accept for lichess-bot)
        self.hash = DF_HASH
        self.threads = DF_THREADS
        self.move_overhead = 200   # milliseconds
        self.ponder = False

        # Tunables (also via UCI options)
        self.sf_depth = DF_SF_DEPTH
        self.sf_multipv = DF_SF_MPVS
        self.expl_topn = DF_EXPL_TOPN
        self.expl_min_games = DF_EXPL_MING
        self.expl_weight = DF_EXPL_W
        self.cloud_multipv = DF_CLOUD_MPVS
        self.cloud_weight = DF_CLOUD_W

        # Reranker (mini-ML) defaults
        self.use_reranker = True
        self.tie_window_cp = 25
        self.model = _safe_load_model()  # optional pickle (e.g., LogisticRegression/LightGBM)
        try:
            print(f"info string reranker={'model' if self.model else 'fallback'}", flush=True)
        except Exception:
            pass

        # Logging / dataset capture (Phase-1)
        self.game_id = ""
        self.log_path = os.getenv("ALF_LOG_PATH", "")  # empty => daily auto file
        self._ply = 0
        self._log_path_announced = False

        # No-op placeholders (make wrappers happy, keep max strength)
        self.syzygy_path = ""
        self.uci_limit_strength = False
        self.uci_elo = 2850
        self.uci_showwdl = False
        self.uci_analysemode = False
        self.uci_chess960 = False

        # Stockfish
        path = find_stockfish()
        if not path:
            print("info string Stockfish not found", flush=True)
            sys.exit(1)
        self.sf = chess.engine.SimpleEngine.popen_uci(path)
        self.sf.configure({"Threads": self.threads, "Hash": self.hash})
        self.lock = threading.Lock()

    # --------- Reranker feature extraction & scoring ----------
    def _phase(self) -> int:
        # crude phase: count pieces (fewer => endgame)
        return len(self.board.piece_map())

    def _features_for(self, mv: chess.Move, base_cp: int, ex: dict, cl: dict) -> List[float]:
        """Compute a tiny feature vector for (mv, position). Keep super cheap."""
        uci = mv.uci()
        # Explorer
        ex_g = ex_w = ex_d = 0.0
        if ex and "moves" in ex:
            for m in ex["moves"]:
                if m.get("uci") == uci:
                    ex_g = float(m.get("gameCount", 0) or 0.0)
                    ex_w = float(m.get("white", 0) or 0.0)
                    ex_d = float(m.get("draws", 0) or 0.0)
                    break
        ex_tot = max(1.0, ex_w + ex_d + float(ex_g - (ex_w + ex_d)))
        ex_winp = ex_w / ex_tot
        ex_drawp = ex_d / ex_tot
        ex_logg = math.log1p(ex_g)

        # Cloud (best matching PV)
        cl_cp = 0.0
        cl_depth = 0.0
        cl_rank = 0.0
        if cl and "pvs" in cl:
            for i, pv in enumerate(cl["pvs"], start=1):
                s = pv.get("moves", "")
                if s and s.split()[0] == uci:
                    cl_cp = float(pv.get("cp", 0.0) or 0.0)
                    cl_depth = float(pv.get("depth", 0.0) or 0.0)
                    cl_rank = 1.0 / i
                    break

        # Local cheap flags
        is_cap = 1.0 if self.board.is_capture(mv) else 0.0
        gives_chk = 1.0 if self.board.gives_check(mv) else 0.0
        to_center = 1.0 if chess.square_file(mv.to_square) in (3,4) and chess.square_rank(mv.to_square) in (3,4) else 0.0
        phase = float(self._phase())

        # Normalize base_cp roughly to [-1,1] around 100cp scale
        base_norm = max(-2.0, min(2.0, base_cp / 100.0))

        return [
            1.0,            # bias
            base_norm,      # SF centipawn (normalized)
            ex_winp,        # explorer win rate (0..1)
            ex_drawp,       # explorer draw rate
            ex_logg,        # log(1+games)
            cl_cp / 100.0,  # cloud cp (normalized)
            cl_depth / 40.0,# cloud depth (0..~1)
            cl_rank,        # 1/rank
            is_cap, gives_chk, to_center,
            phase / 32.0,   # coarse phase scale
        ]

    def _score_with_model(self, feats: List[float]) -> float:
        """Return a scalar score. If model exists, use it; else tiny linear fallback."""
        if self.model is not None:
            try:
                if hasattr(self.model, "predict_proba"):
                    p = float(self.model.predict_proba([feats])[0][-1])
                    return p
                if hasattr(self.model, "decision_function"):
                    z = float(self.model.decision_function([feats])[0])
                    return _sigmoid(z)
                if hasattr(self.model, "predict"):
                    y = float(self.model.predict([feats])[0])
                    return y
            except Exception:
                pass
        # Fallback linear “tiny brain” (hand-tuned but ML-shaped)
        w = [  # same length as feats
            0.0,     # bias
            0.9,     # base_norm (trust SF)
            0.4,     # ex_winp
            -0.1,    # ex_drawp (slightly prefer less drawish)
            0.15,    # ex_logg (trust big samples)
            0.25,    # cloud cp
            0.2,     # cloud depth
            0.15,    # 1/rank
            0.05,    # is_cap
            0.05,    # gives_check
            0.04,    # to_center
            -0.05,   # phase (prefer simpler in endgame if close)
        ]
        z = sum(a*b for a,b in zip(w, feats))
        return _sigmoid(z)

    # ----------------- Logging helpers -----------------
    def _effective_log_path(self) -> str:
        return self.log_path or f"alf_log_{datetime.date.today().isoformat()}.csv"

    def _log_csv_row(self, row: dict):
        try:
            path = self._effective_log_path()
            need_header = not os.path.exists(path)
            # Announce once where we're logging
            if not self._log_path_announced:
                try:
                    print(f"info string logging to {path}", flush=True)
                except Exception:
                    pass
                self._log_path_announced = True
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "ts","game_id","ply","fen","turn",
                    "chosen_uci","mode","sf_best_cp","chosen_sf_cp","tie_window_cp",
                    "explorer_used","cloud_used","candidates_json"
                ])
                if need_header:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:
            try:
                print(f"info string log_csv_error={e}", flush=True)
            except Exception:
                pass

    # ----------------- UCI protocol -----------------
    def uci(self):
        print("id name Mobius-Grey ALF v3", flush=True)
        print("id author chris", flush=True)

        # Required/common options
        print(f"option name Hash type spin default {DF_HASH} min 16 max 4096", flush=True)
        print(f"option name Threads type spin default {DF_THREADS} min 1 max 32", flush=True)
        print("option name Move Overhead type spin default 200 min 0 max 10000", flush=True)
        print("option name Ponder type check default false", flush=True)

        # Tunables
        print(f"option name SFDepth type spin default {DF_SF_DEPTH} min 6 max 40", flush=True)
        print(f"option name SFMultiPV type spin default {DF_SF_MPVS} min 1 max 16", flush=True)
        print(f"option name ExplorerTopN type spin default {DF_EXPL_TOPN} min 5 max 100", flush=True)
        print(f"option name ExplorerMinGames type spin default {DF_EXPL_MING} min 0 max 1000", flush=True)
        print(f"option name ExplorerWeight type spin default {DF_EXPL_W:.0f} min 0 max 200", flush=True)
        print(f"option name CloudMultiPV type spin default {DF_CLOUD_MPVS} min 1 max 16", flush=True)
        print(f"option name CloudWeight type spin default {DF_CLOUD_W:.0f} min 0 max 200", flush=True)

        # Reranker options
        print("option name UseReranker type check default true", flush=True)
        print("option name TieWindowCP type spin default 25 min 0 max 200", flush=True)

        # Logging / metadata
        print("option name GameId type string default ", flush=True)
        print(f"option name LogPath type string default {self.log_path or 'alf_log_YYYY-MM-DD.csv'}", flush=True)

        # Compatibility no-ops
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
        self.board.reset()
        self._ply = 0

    def setoption(self, name: str, value: str):
        key = name.strip().lower().replace(" ", "")
        try:
            if key == "hash":
                self.hash = int(value); self.sf.configure({"Hash": self.hash})
            elif key == "threads":
                self.threads = int(value); self.sf.configure({"Threads": self.threads})
            elif key == "moveoverhead":
                self.move_overhead = int(value)
            elif key == "ponder":
                self.ponder = str(value).lower() == "true"
            elif key == "sfdepth":
                self.sf_depth = int(value)
            elif key == "sfmultipv":
                self.sf_multipv = max(1, int(value))
            elif key == "explorertopn":
                self.expl_topn = max(1, int(value))
            elif key == "explorermingames":
                self.expl_min_games = max(0, int(value))
            elif key == "explorerweight":
                self.expl_weight = max(0.0, float(value))
            elif key == "cloudmultipv":
                self.cloud_multipv = max(1, int(value))
            elif key == "cloudweight":
                self.cloud_weight = max(0.0, float(value))
            elif key == "usereranker":
                self.use_reranker = (str(value).lower() == "true")
            elif key == "tiewindowcp":
                self.tie_window_cp = max(0, int(value))
            elif key == "gameid":
                self.game_id = value or ""
            elif key == "logpath":
                self.log_path = value or ""
                # Reset announce so we re-print if path changes
                self._log_path_announced = False
            elif key == "syzygypath":
                self.syzygy_path = value or ""
                try:
                    self.sf.configure({"SyzygyPath": self.syzygy_path})
                except Exception:
                    pass
            elif key == "uci_limitstrength":
                self.uci_limit_strength = False
            elif key == "uci_elo":
                self.uci_elo = 2850
            elif key == "uci_showwdl":
                self.uci_showwdl = str(value).lower() == "true"
            elif key == "uci_analysemode":
                self.uci_analysemode = False
            elif key == "uci_chess960":
                self.uci_chess960 = False
        except Exception:
            pass

    # ---- Position / search logic ----
    def position(self, tokens: List[str]):
        if not tokens: return
        if tokens[0] == "startpos":
            self.board.reset(); self._ply = 0
            if len(tokens) >= 3 and tokens[1] == "moves":
                for u in tokens[2:]:
                    try:
                        self.board.push_uci(u); self._ply += 1
                    except Exception:
                        break
        elif tokens[0] == "fen":
            fen = " ".join(tokens[1:7])
            try:
                self.board.set_fen(fen)
                # Conservative: reset ply on manual FEN; will rebuild from subsequent moves
                self._ply = 0
            except Exception:
                return
            tail = tokens[7:]
            if tail and tail[0] == "moves":
                for u in tail[1:]:
                    try:
                        self.board.push_uci(u); self._ply += 1
                    except Exception:
                        break

    def _should_query_online(self, movetime_ms: Optional[int], wtime: Optional[int], btime: Optional[int]) -> bool:
        # Skip online calls in tiny movetime or severe time pressure
        if movetime_ms is not None and movetime_ms < 200:
            return False
        if wtime is not None and btime is not None:
            side_ms = wtime if self.board.turn == chess.WHITE else btime
            if side_ms < 3000:
                return False
        return True

    def sf_heads(self, movetime_ms: Optional[int], wtime: Optional[int], btime: Optional[int],
                 winc: int, binc: int) -> List[Tuple[chess.Move, int]]:
        if movetime_ms is not None:
            # Apply Move Overhead
            oh = max(0.0, self.move_overhead)
            tm_ms = max(10.0, movetime_ms - oh)
            limit = chess.engine.Limit(time=tm_ms / 1000.0)
        elif wtime is not None and btime is not None:
            side = self.board.turn
            rem_ms = float(wtime if side == chess.WHITE else btime)
            inc_ms = float(winc if side == chess.WHITE else binc)
            oh = max(0.0, float(self.move_overhead))
            t_ms = max(40.0, min(5000.0, rem_ms / 30.0 + inc_ms - oh))
            limit = chess.engine.Limit(time=t_ms / 1000.0)
        else:
            limit = chess.engine.Limit(depth=self.sf_depth)

        with self.lock:
            info_list = self.sf.analyse(self.board, limit, multipv=self.sf_multipv)

        heads: List[Tuple[chess.Move, int]] = []
        seen = set()
        wt = self.board.turn
        for inf in info_list:
            pv = inf.get("pv")
            if not pv: continue
            mv = pv[0]
            if mv in seen: continue
            seen.add(mv)
            cp = cp_from_score(inf["score"], wt)
            heads.append((mv, cp))

        if not heads:
            try:
                m0 = next(iter(self.board.legal_moves))
                heads = [(m0, 0)]
            except StopIteration:
                heads = []
        return heads

    def pick_move(self, movetime_ms: Optional[int], wtime: Optional[int], btime: Optional[int],
                  winc: int, binc: int) -> Optional[chess.Move]:
        heads = self.sf_heads(movetime_ms, wtime, btime, winc, binc)
        if not heads:
            return None

        fen = self.board.fen()
        ex = cl = None
        if (self.expl_weight > 0 or self.cloud_weight > 0) and self._should_query_online(movetime_ms, wtime, btime):
            ex = explorer_request_cached(fen, self.expl_topn)
            cl = cloud_request_cached(fen, self.cloud_multipv)

        # Baseline: compute blended scores
        blended_scores: List[Tuple[chess.Move, int, float]] = []
        best_mv, best_score = None, -1e18
        for mv, base_cp in heads:
            uci = mv.uci()
            score = float(base_cp)
            if ex:
                score += explorer_bonus_for(uci, ex, self.expl_min_games, self.expl_weight)
            if cl:
                score += cloud_bonus_for(uci, cl, self.cloud_weight)
            if self.board.is_capture(mv) or self.board.gives_check(mv):
                score += 2.0
            blended_scores.append((mv, base_cp, score))
            if score > best_score:
                best_score, best_mv = score, mv

        if not self.use_reranker or len(blended_scores) <= 1:
            chosen = best_mv or heads[0][0]
            why = "no_ml"
        else:
            # Rerank only among near-equals by SF base cp
            sf_best = max(cp for _, cp, _ in blended_scores)
            window = self.tie_window_cp
            candidates = [(mv, base_cp, score) for (mv, base_cp, score) in blended_scores
                          if base_cp >= sf_best - window]
            if not candidates:
                chosen = best_mv or heads[0][0]
                why = "no_candidates"
            else:
                best_mv_ml, best_s = None, -1e9
                for mv, base_cp, _ in candidates:
                    feats = self._features_for(mv, base_cp, ex or {}, cl or {})
                    s = self._score_with_model(feats)
                    if s > best_s:
                        best_s, best_mv_ml = s, mv
                # Failsafe: don't take a move far worse than SF best
                picked_cp = next(cp for m, cp, _ in blended_scores if m == best_mv_ml)
                if picked_cp < sf_best - 100:
                    chosen = best_mv
                    why = "fallback_sf"
                else:
                    chosen = best_mv_ml
                    why = "rerank"
                try:
                    print(f"info string pick={chosen.uci()} sf_best={sf_best} picked_cp={picked_cp} mode={why} window={window} ml={best_s:.3f}", flush=True)
                except Exception:
                    pass

        # --- Phase-1 ML logging (CSV) ---
        try:
            # Compute sf_best for logging even if no ML path
            sf_best_for_log = max(cp for _, cp, _ in blended_scores) if blended_scores else None
            chosen_sf_cp = None
            chosen_mv = chosen or best_mv or heads[0][0]
            for m, cp, _ in blended_scores:
                if m == chosen_mv:
                    chosen_sf_cp = cp
                    break
            cand_rows = []
            for mv, base_cp, blend in blended_scores:
                feats = self._features_for(mv, base_cp, ex or {}, cl or {})
                cand_rows.append({"uci": mv.uci(), "sf_cp": base_cp, "blend": blend, "feats": feats})

            row = {
                "ts": time.time(),
                "game_id": self.game_id,
                "ply": self._ply + 1,  # about to play this move
                "fen": self.board.fen(),
                "turn": "w" if self.board.turn else "b",
                "chosen_uci": chosen_mv.uci() if chosen_mv else None,
                "mode": why,
                "sf_best_cp": sf_best_for_log,
                "chosen_sf_cp": chosen_sf_cp,
                "tie_window_cp": self.tie_window_cp,
                "explorer_used": 1 if ex else 0,
                "cloud_used": 1 if cl else 0,
                "candidates_json": json.dumps(cand_rows, separators=(",", ":")),
            }
            self._log_csv_row(row)
        except Exception as e:
            try:
                print(f"info string log_pack_error={e}", flush=True)
            except Exception:
                pass

        return chosen or best_mv or heads[0][0]

    def go_and_print(self, cmd: str):
        parts = cmd.split()
        def gi(tag): return int(parts[parts.index(tag)+1]) if tag in parts else None
        movetime = gi("movetime")
        wtime    = gi("wtime")
        btime    = gi("btime")
        winc     = gi("winc") or 0
        binc     = gi("binc") or 0

        mv = self.pick_move(movetime, wtime, btime, winc, binc)
        if mv is None:
            print("bestmove 0000", flush=True)
        else:
            print("bestmove " + mv.uci(), flush=True)

    def loop(self):
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            if line == "uci":
                self.uci()
            elif line == "isready":
                self.isready()
            elif line == "ucinewgame":
                self.ucinewgame()
            elif line.startswith("setoption"):
                try:
                    parts = shlex.split(line)
                except Exception:
                    parts = line.split()
                name, value, seen_value = [], None, False
                for tok in parts[1:]:
                    if tok == "name": continue
                    if tok == "value":
                        seen_value = True
                        continue
                    if seen_value:
                        value = tok if value is None else value + " " + tok
                    else:
                        name.append(tok)
                self.setoption(" ".join(name), value if value is not None else "")
            elif line.startswith("position"):
                self.position(line.split()[1:])
            elif line.startswith("go"):
                self.go_and_print(line)
            elif line == "stop":
                pass
            elif line == "quit":
                break
        try:
            self.sf.close()
        except Exception:
            pass

# --------------------- entry -------------------------
if __name__ == "__main__":
    ALFEngine().loop()
