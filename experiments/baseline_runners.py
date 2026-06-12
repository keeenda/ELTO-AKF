"""
baseline_runners.py
===================
Shared module providing run_spkfnui() for use in both
lv_elto_eval.py and lorenz96.py.

Extracts SPKFnUI architecture + train_model + eval_model from
neural_kf_baseline.py and wraps them into a single callable that
accepts pre-split numpy arrays and returns (test_mse, wall_time_s).
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# SPKFnUI model  (Loo et al., IEEE TSMC 2024)
# ══════════════════════════════════════════════════════════════════════════════

class SPKFnUI(nn.Module):
    """
    Sigma-Point Kalman Filter with Nonlinear Unknown-Input estimation.
    Single GRU φ(x̂_t) using POSTERIOR state as input (paper's key claim).
    """

    def __init__(self, obs_dim: int, state_dim: int, gru_hidden: int = 32):
        super().__init__()
        self.obs_dim   = obs_dim
        self.state_dim = state_dim
        self.ui_dim    = state_dim
        self.h         = gru_hidden

        self.F       = nn.Parameter(torch.eye(state_dim))
        self.G       = nn.Parameter(0.1 * torch.randn(state_dim, state_dim))
        H = torch.zeros(obs_dim, state_dim)
        H[:min(obs_dim, state_dim), :min(obs_dim, state_dim)] = \
            torch.eye(min(obs_dim, state_dim))
        self.register_buffer("H", H)
        self.log_q   = nn.Parameter(torch.full((state_dim,), -2.0))
        self.log_r   = nn.Parameter(torch.full((obs_dim,),   -1.0))

        self.gru_phi = nn.GRU(state_dim, gru_hidden, batch_first=True)
        self.fc_u    = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden), nn.Tanh(),
            nn.Linear(gru_hidden, state_dim),
        )
        self.fc_log_eps = nn.Linear(gru_hidden, state_dim)

    def _Q(self): return torch.diag(torch.exp(self.log_q))
    def _R(self): return torch.diag(torch.exp(self.log_r))

    def forward(self, observations: torch.Tensor):
        T, dev  = observations.shape[0], observations.device
        F, G, H = self.F, self.G, self.H
        Q, R    = self._Q(), self._R()
        n, m    = self.state_dim, self.obs_dim

        h_phi = torch.zeros(1, 1, self.h, device=dev)
        x_t   = torch.zeros(n, 1, device=dev)
        P_t   = torch.eye(n, device=dev)

        x_seq = []
        for t in range(T):
            z_t = observations[t].unsqueeze(-1)

            # ── Update (correction) ───────────────────────────────────────────
            innov  = z_t - H @ x_t
            S_inn  = H @ P_t @ H.T + R
            K_t    = P_t @ H.T @ torch.inverse(
                S_inn + 1e-6 * torch.eye(m, device=dev))
            x_post = x_t + K_t @ innov
            IKH    = torch.eye(n, device=dev) - K_t @ H
            P_post = IKH @ P_t @ IKH.T + K_t @ R @ K_t.T

            # ── UI estimation from POSTERIOR state ────────────────────────────
            gru_in         = x_post.squeeze(-1).unsqueeze(0).unsqueeze(0)
            gru_out, h_phi = self.gru_phi(gru_in, h_phi)
            hidden         = gru_out.squeeze(0).squeeze(0)
            u_hat          = self.fc_u(hidden).unsqueeze(-1)
            E_t            = torch.diag(torch.exp(self.fc_log_eps(hidden)))

            # ── Prediction with joint covariance ─────────────────────────────
            M_t     = torch.eye(self.ui_dim, n, device=dev)
            J_t     = F + G @ M_t
            x_prior = F @ x_post + G @ u_hat
            P_prior = J_t @ P_post @ J_t.T + G @ E_t @ G.T + Q

            x_t = x_prior
            P_t = P_prior
            x_seq.append(x_post.squeeze(-1))

        return torch.stack(x_seq, dim=0)   # (T, n)


# ══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    d = min(pred.shape[1], target.shape[1])
    return torch.mean((pred[:, :d] - target[:, :d]) ** 2)


def _train(model, train_obs, train_gt, val_obs, val_gt,
           lr=1e-3, max_epochs=300, patience=30, l2=1e-4):
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=10, min_lr=1e-5)

    best_val, best_sd, no_imp = float("inf"), None, 0

    # Fallback: save initial weights so best_sd is never None
    model.eval()
    with torch.no_grad():
        try:    v0 = _mse(model(val_obs), val_gt).item()
        except: v0 = float("inf")
    if v0 == v0:   # not NaN
        best_val, best_sd = v0, copy.deepcopy(model.state_dict())

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        try:
            loss = _mse(model(train_obs), train_gt)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            l2_reg = sum(p.pow(2).sum() for p in model.parameters())
            (loss + l2 * l2_reg).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        except Exception:
            continue

        model.eval()
        with torch.no_grad():
            try:    val_loss = _mse(model(val_obs), val_gt).item()
            except: val_loss = float("inf")

        if val_loss != val_loss or val_loss == float("inf"):
            continue

        sched.step(val_loss)
        if val_loss < best_val - 1e-8:
            best_val, best_sd, no_imp = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_sd is None:
        best_sd = copy.deepcopy(model.state_dict())
    return best_val, best_sd


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def run_spkfnui(train_input, val_input, val_gt, test_input, test_gt,
                gru_hidden=32, lr=1e-3, max_epochs=300,
                patience=30, l2=1e-4, model_seed=None):
    """
    Train and evaluate SPKF-nUI on pre-split numpy arrays.

    Parameters
    ----------
    train_input : (T_train, obs_dim)   noisy observations for training
    val_input   : (T_val,   obs_dim)   noisy observations for validation
    val_gt      : (T_val,   state_dim) clean groundtruth for val
    test_input  : (T_test,  obs_dim)   noisy observations for test
    test_gt     : (T_test,  state_dim) clean groundtruth for test
    gru_hidden  : GRU hidden size (keep small for short sequences)

    Returns
    -------
    (test_mse, wall_time_s)
    """
    if model_seed is not None:
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)

    def t(a):
        a = np.array(a, dtype=np.float32)
        if a.ndim == 1: a = a[:, None]
        return torch.tensor(a).to(DEVICE)

    obs_dim   = train_input.shape[1] if train_input.ndim == 2 else 1
    state_dim = test_gt.shape[1]     if test_gt.ndim     == 2 else 1

    # Proxy GT for training: use first state_dim cols of noisy obs
    used          = min(obs_dim, state_dim)
    train_gt_t    = t(train_input[:, :used])
    if used < state_dim:
        pad        = torch.zeros(len(train_input), state_dim - used, device=DEVICE)
        train_gt_t = torch.cat([train_gt_t, pad], dim=1)

    model = SPKFnUI(obs_dim, state_dim, gru_hidden).to(DEVICE)

    t_start = time.time()

    _, best_sd = _train(
        model,
        train_obs = t(train_input),
        train_gt  = train_gt_t,
        val_obs   = t(val_input),
        val_gt    = t(val_gt),
        lr=lr, max_epochs=max_epochs, patience=patience, l2=l2,
    )

    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        test_mse = _mse(model(t(test_input)), t(test_gt)).item()

    wall_time = time.time() - t_start
    return test_mse, wall_time
