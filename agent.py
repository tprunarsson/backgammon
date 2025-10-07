#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch DQN agent for Backgammon (after-state evaluation).

Public API used by train.py / tournament:
  - action(board, dice, player, i, train=False, train_config=None)
  - game_over_update(board, reward)
  - set_eval_mode(is_eval)
  - save(path="checkpoints/best.pt")
  - load(path="checkpoints/best.pt")

This agent evaluates ALL legal after-states for the current dice roll,
chooses ε-greedy during training, greedy in eval/competition.
"""

from collections import deque
from pathlib import Path
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import backgammon  # your Numba-optimized core

# ---------------- Config ----------------
class Config:
    # Features: 24 points + 4 extras (bars/off) + 1 "moves_left"
    state_dim = 24 + 4 + 1
    gamma = 0.99
    lr = 1e-3
    batch_size = 256
    buffer_size = 100_000
    start_learning_after = 2_000
    target_update_every = 2_000
    train_every = 1
    eps_start = 0.10
    eps_end = 0.01
    eps_decay = 200_000  # steps from start -> end
    hidden = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

CFG = Config()

# ------------- Flip helpers (local) -------------
_FLIP_IDX = np.array(
    [0, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
     12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 26, 25, 28, 27],
    dtype=np.int32
)

def _flip_board(board):
    out = np.empty(29, dtype=board.dtype)
    out[:] = -board[_FLIP_IDX]
    return out

def _flip_move(move):
    if len(move) == 0:
        return move
    mv = np.asarray(move, dtype=np.int32).copy()
    for r in range(mv.shape[0]):
        mv[r, 0] = _FLIP_IDX[mv[r, 0]]
        mv[r, 1] = _FLIP_IDX[mv[r, 1]]
    return mv

# ------------- Replay Buffer -------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, s, r, s_next, done):
        self.buf.append((s, r, s_next, done))
    def __len__(self):
        return len(self.buf)
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, r, s2, d = zip(*batch)
        return (
            torch.as_tensor(np.stack(s), dtype=torch.float32, device=CFG.device),
            torch.as_tensor(r, dtype=torch.float32, device=CFG.device).unsqueeze(1),
            torch.as_tensor(np.stack(s2), dtype=torch.float32, device=CFG.device),
            torch.as_tensor(d, dtype=torch.bool, device=CFG.device).unsqueeze(1),
        )

# ------------- Model -------------
class QNet(nn.Module):
    def __init__(self, in_dim=CFG.state_dim, hid=CFG.hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1),
        )
    def forward(self, x):
        return self.net(x)

_qnet = QNet().to(CFG.device)
_tnet = QNet().to(CFG.device)
_tnet.load_state_dict(_qnet.state_dict())
_opt = torch.optim.Adam(_qnet.parameters(), lr=CFG.lr)
_buf = ReplayBuffer(CFG.buffer_size)

_steps = 0
_epsilon = CFG.eps_start
_eval_mode = False

# ------------- Save / Load -------------
CHECKPOINT_PATH = Path("checkpoints/best.pt")
_loaded_from_disk = False

def save(path: str = str(CHECKPOINT_PATH)):
    """Save current Q-network to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"qnet": _qnet.state_dict()}, p)

def load(path: str = str(CHECKPOINT_PATH), map_location: str | torch.device = "cpu"):
    """Load Q-network weights; sync target; enter eval mode."""
    global _loaded_from_disk
    state = torch.load(path, map_location=map_location)
    _qnet.load_state_dict(state["qnet"])
    _tnet.load_state_dict(_qnet.state_dict())
    set_eval_mode(True)
    _loaded_from_disk = True

def _lazy_load_if_available():
    """Auto-load once at first eval-time action if checkpoint exists."""
    global _loaded_from_disk
    if _loaded_from_disk:
        return
    if CHECKPOINT_PATH.exists():
        try:
            load(str(CHECKPOINT_PATH), map_location="cpu")
        except Exception:
            pass
        _loaded_from_disk = True

# ------------- Features -------------
def _moves_left(dice, i):
    # In this codebase: doubles -> two applications; otherwise one.
    return 1 + int(dice[0] == dice[1]) - i

def _encode_state(board_flipped, moves_left):
    """
    board_flipped: +1 perspective
    features: 24 points, bar_self, bar_opp, off_self, off_opp, moves_left
    """
    x = np.zeros(CFG.state_dim, dtype=np.float32)
    x[:24] = board_flipped[1:25] * 0.2
    x[24]  = board_flipped[25] * 0.2
    x[25]  = board_flipped[26] * 0.2
    x[26]  = board_flipped[27] / 15.0
    x[27]  = board_flipped[28] / 15.0
    x[28]  = float(moves_left)
    return x

def _bearing_off_win(board_flipped):
    # +1 has all men off?
    return board_flipped[27] == 15

# ------------- Learning -------------
def _update_epsilon():
    global _epsilon
    frac = min(1.0, _steps / CFG.eps_decay)
    _epsilon = CFG.eps_start + (CFG.eps_end - CFG.eps_start) * frac
    if _eval_mode:
        _epsilon = 0.0

def _maybe_learn():
    global _steps
    if _eval_mode: return
    if len(_buf) < CFG.start_learning_after: return
    if _steps % CFG.train_every != 0: return

    S, R, S2, D = _buf.sample(CFG.batch_size)
    with torch.no_grad():
        Q2 = _tnet(S2)                          # bootstrap from s'
        target = R + (~D) * (CFG.gamma * Q2)    # zero if done

    Q = _qnet(S)
    loss = F.mse_loss(Q, target)

    _opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(_qnet.parameters(), 1.0)
    _opt.step()

    if _steps % CFG.target_update_every == 0:
        _tnet.load_state_dict(_qnet.state_dict())

# ------------- Hooks -------------
def set_eval_mode(is_eval: bool):
    global _eval_mode
    _eval_mode = bool(is_eval)

def episode_start():
    pass

def end_episode(outcome, final_board, perspective):
    pass

def game_over_update(board, reward):
    """
    Trainer calls this at episode end (twice: normal & flipped).
    Push a terminal sample: (s, r, s, done=True).
    """
    s = _encode_state(board, moves_left=0)
    _buf.push(s, float(reward), s, True)

# ------------- Policy -------------
def action(board_copy, dice, player, i, train=False, train_config=None):
    """
    Returns [] if no legal moves, else an array of shape (k,2) of [start,end] moves.
    Also performs learning if train=True.
    """
    global _steps

    if not train:
        _lazy_load_if_available()  # load competition weights if present

    # Act from +1 perspective
    board_pov = _flip_board(board_copy) if player == -1 else board_copy

    # Enumerate after-states
    possible_moves, possible_boards = backgammon.legal_moves(board_pov, dice, player=1)
    nA = len(possible_moves)
    if nA == 0:
        return []

    moves_left = 1 + int(dice[0] == dice[1]) - i
    S = _encode_state(board_pov, moves_left)
    S_primes = np.stack([_encode_state(b, moves_left - 1) for b in possible_boards], axis=0)

    S_t  = torch.as_tensor(S[None, :], dtype=torch.float32, device=CFG.device)
    Sp_t = torch.as_tensor(S_primes, dtype=torch.float32, device=CFG.device)

    # Evaluate after-states WITHOUT tracking grads (selection only)
    with torch.no_grad():
        q_asp = _qnet(Sp_t).squeeze(1)  # [nA]

    # ε-greedy
    if train and (not _eval_mode) and (random.random() < _epsilon):
        a_idx = random.randrange(nA)
    else:
        a_idx = int(torch.argmax(q_asp).item())

    chosen_move = possible_moves[a_idx]
    chosen_board = possible_boards[a_idx]

    # reward & transition
    r = 1.0 if (chosen_board[27] == 15) else 0.0
    done = bool(r > 0.0)

    if train and (not _eval_mode):
        _buf.push(S, r, S_primes[a_idx], done)
        _steps += 1
        _update_epsilon()
        _maybe_learn()  # <-- now runs with autograd enabled

    # map back if needed
    if player == -1:
        chosen_move = _flip_move(chosen_move)
    return chosen_move
