#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALF — Explorer + Cloud Eval + Stockfish blended UCI engine

- Answers UCI (uci/isready/ucinewgame/position/go/stop/quit)
- Uses Stockfish MultiPV as a base
- Adds Explorer (book) EV and Cloud Eval bonuses to candidate moves
- Clean fallback to pure Stockfish
- Accepts lichess-bot's common options (Hash, Threads, Move Overhead, Ponder)
- Extra tunables exposed as UCI options (ExplorerWeight, CloudWeight, MinGames, SFDepth, SFMultiPV, CloudMultiPV, ExplorerTopN)
- NEW: Advertises SyzygyPath / UCI_LimitStrength / UCI_Elo / UCI_ShowWDL / UCI_AnalyseMode / UCI_Chess960
       as no-ops so wrappers won't crash.

Requires:
    pip install python-chess requests
    sudo apt install stockfish   (or set env STOCKFISH_PATH)
"""

import os
import sys
import shlex
import shutil
import threading
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

# --------------------- online data --------------------
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
        self.move_overhead = 200
        self.ponder = False

        # Tunables (also via UCI options)
        self.sf_depth = DF_SF_DEPTH
        self.sf_multipv = DF_SF_MPVS
        self.expl_topn = DF_EXPL_TOPN
        self.expl_min_games = DF_EXPL_MING
        self.expl_weight = DF_EXPL_W
        self.cloud_multipv = DF_CLOUD_MPVS
        self.cloud_weight = DF_CLOUD_W

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

    # ---- UCI protocol ----
    def uci(self):
        print("id name ALF ExplorerCloudBlend", flush=True)
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
            elif key == "syzygypath":
                self.syzygy_path = value or ""
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

    # ---- Position / search logic (unchanged) ----
    def position(self, tokens: List[str]):
        if not tokens: return
        if tokens[0] == "startpos":
            self.board.reset()
            if len(tokens) >= 3 and tokens[1] == "moves":
                for u in tokens[2:]:
                    try: self.board.push_uci(u)
                    except Exception: break
        elif tokens[0] == "fen":
            fen = " ".join(tokens[1:7])
            try: self.board.set_fen(fen)
            except Exception: return
            tail = tokens[7:]
            if tail and tail[0] == "moves":
                for u in tail[1:]:
                    try: self.board.push_uci(u)
                    except Exception: break

    def sf_heads(self, movetime_ms: Optional[int], wtime: Optional[int], btime: Optional[int],
                 winc: int, binc: int) -> List[Tuple[chess.Move, int]]:
        if movetime_ms is not None:
            limit = chess.engine.Limit(time=max(0.01, movetime_ms / 1000.0))
        elif wtime is not None and btime is not None:
            side = self.board.turn
            rem = (wtime if side == chess.WHITE else btime) / 1000.0
            inc = (winc if side == chess.WHITE else binc) / 1000.0
            t = max(0.04, min(5.0, rem / 30.0 + inc))
            limit = chess.engine.Limit(time=t)
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
        ex = explorer_request(fen, self.expl_topn)
        cl = cloud_request(fen, self.cloud_multipv)

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
            if score > best_score:
                best_score, best_mv = score, mv
        return best_mv or heads[0][0]

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
