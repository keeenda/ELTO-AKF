"""
lorenz96_elto_eval.py
=====================
Lorenz-96 scalability experiment for ELTO-KF vs ELTO-AKF.

Part 1  ── Lorenz-96 simulator
    Integrates the Lorenz-96 ODE with scipy RK45.
    data_seed controls IC + obs noise (fixed → reproducible data).
    Saves one .npz per dimension D in a temp directory.

Part 2  ── Evaluation loop  (mirrors ablation_trajectory_unified.py)
    Runs ELTO-KF and ELTO-AKF on each generated .npz.
    Records MSE + total wall-clock time per run.
    Writes a single structured .log file.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER-EDITABLE CONSTANTS  (top of main — change here, not via CLI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DIM_LIST    = [5, 10, 20, 40]   # state dimensions to sweep
    T_STEPS     = 1000              # integration steps per run
    F_FORCING   = 8.0               # Lorenz-96 forcing constant
    OBS_NOISE   = 0.1               # Gaussian obs noise std
    DATA_SEED   = 42                # fixed — reproducible data
    MODEL_SEED  = None              # None = not fixed; set int for repro

    WINDOW_SIZE = 5                 # ELTO window size
    CMA_ITERS   = 10                # CMA-ES iterations
    EPOCHS      = 200               # spectral learning epochs
    BATCH_SIZE  = 50
    ALPHA       = 0.9               # AKF memory factor (Q)
    BETA        = 0.9               # AKF memory factor (R)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    python lorenz96_elto_eval.py
    python lorenz96_elto_eval.py --data-seed 7 --model-seed 123
    python lorenz96_elto_eval.py --logdir ./logs
"""

import argparse
import logging
import os
import sys
import time
import tempfile
from datetime import datetime

import numpy as np
import cma
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler

# ── ELTO imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.oELTO_eval import OELTO
from model.utils import (parameter_naming, parameter_transform,
                         exception_catcher, parameter_arrays)
from baseline_runners import run_spkfnui


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Lorenz-96 Simulator
# ══════════════════════════════════════════════════════════════════════════════

def lorenz96_rhs(t, x, F):
    """
    Lorenz-96 right-hand side:
        dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    Indices are cyclic (mod D).
    """
    D = len(x)
    dx = np.empty(D)
    for i in range(D):
        dx[i] = (x[(i + 1) % D] - x[(i - 2) % D]) * x[(i - 1) % D] - x[i] + F
    return dx


def simulate_lorenz96(D, T_steps, F=8.0, dt=0.01, data_seed=42, obs_noise=0.1):
    """
    Simulate Lorenz-96 and return train/val/test splits as numpy arrays.

    Returns
    -------
    npz_dict : dict with keys matching the .npz convention used by
               run_single_experiment():
                 'train_data'       : (T_train, D)  noisy observations
                 'test_obs'         : (T_test,  D)  noisy observations
                 'test_groundtruth' : (T_test,  D)  clean states
    """
    rng = np.random.default_rng(data_seed)

    # ── Initial condition: small perturbation around equilibrium ─────────────
    x0 = F * np.ones(D)
    x0[0] += 0.01          # classic Lorenz-96 IC perturbation

    # ── Spin-up: discard first 500 steps to reach attractor ──────────────────
    t_spinup = np.linspace(0, 500 * dt, 500)
    sol_spinup = solve_ivp(lorenz96_rhs, [t_spinup[0], t_spinup[-1]],
                           x0, args=(F,), method='RK45',
                           t_eval=t_spinup, rtol=1e-8, atol=1e-8)
    x_start = sol_spinup.y[:, -1]

    # ── Main integration ──────────────────────────────────────────────────────
    t_eval = np.linspace(0, T_steps * dt, T_steps)
    sol = solve_ivp(lorenz96_rhs, [t_eval[0], t_eval[-1]],
                    x_start, args=(F,), method='RK45',
                    t_eval=t_eval, rtol=1e-8, atol=1e-8)

    # sol.y : (D, T_steps)  →  clean states (T_steps, D)
    clean = sol.y.T.astype(np.float32)

    # ── Standardise clean states before adding noise ──────────────────────────
    scaler = StandardScaler()
    clean_scaled = scaler.fit_transform(clean)

    # ── Add Gaussian observation noise ────────────────────────────────────────
    noise = rng.normal(0.0, obs_noise, size=clean_scaled.shape).astype(np.float32)
    noisy = (clean_scaled + noise).astype(np.float32)

    # ── 50 / 25 / 25 split (mirrors oelto_pen.py convention) ─────────────────
    n      = T_steps
    i_mid  = n // 2        # end of train
    i_3q   = n * 3 // 4    # end of val (= start of test)

    train_data       = noisy[:i_mid]
    test_obs         = noisy[i_mid:]          # val + test concatenated
    test_groundtruth = clean_scaled[i_mid:]   # ground truth for val+test

    return {
        'train_data':       train_data,
        'test_obs':         test_obs,
        'test_groundtruth': test_groundtruth,
    }


def save_lorenz96_npz(D, T_steps, F, obs_noise, data_seed, out_dir):
    """Simulate and save to .npz; return the file path."""
    npz_dict = simulate_lorenz96(D, T_steps, F=F, data_seed=data_seed,
                                  obs_noise=obs_noise)
    fname = f"lorenz96_D{D}_T{T_steps}_F{int(F)}_dseed{data_seed}.npz"
    fpath = os.path.join(out_dir, fname)
    np.savez(fpath, **npz_dict)
    return fpath


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Evaluation (mirrors ablation_trajectory_unified.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_single_experiment(dataset_path, noise_model,
                          window_size=5, k_val=5,
                          cma_iters=10, epochs=200, batch_size=50,
                          alpha=0.9, beta=0.9,
                          model_seed=None):
    """
    Load .npz → train ELTO → CMA-ES → test.
    Returns (test_mse, total_wall_time_s).
    Total wall time = operator training + CMA-ES + inference (end-to-end).
    """
    if model_seed is not None:
        np.random.seed(model_seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    raw          = np.load(dataset_path)
    train_input  = raw['train_data'].astype(np.float32)
    test_obs     = raw['test_obs'].astype(np.float32)
    test_gt      = raw['test_groundtruth'].astype(np.float32)

    def ensure2d(a):
        return a[:, None] if a.ndim == 1 else a

    train_input = ensure2d(train_input)
    test_obs    = ensure2d(test_obs)
    test_gt     = ensure2d(test_gt)

    # val = first half of test_obs, test = second half
    split             = len(test_obs) // 2
    val_input         = test_obs[:split]
    val_gt            = test_gt[:split]
    final_test_input  = test_obs[split:]
    final_test_gt     = test_gt[split:]

    # ── Hyper-params ──────────────────────────────────────────────────────────
    eps_t, eps_o, eps_q, eps_l, eps_m = -5, -6, -10, -3, -3

    if noise_model == 'static':
        all_param_names  = ['eps_t', 'eps_o', 'eps_q']
        x_0              = [eps_t, eps_o, eps_q]
        decorator_kwargs = {}
        model_config     = {}

    elif noise_model == 'fixed_bws':
        # SB-structured Q/R, optimised offline, NO dynamic updates.
        # Semantically equivalent to alpha=beta=1 AKF but without the
        # decompose-on-every-window call that causes eigh to blow up.
        k_q = k_r = k_val
        num_params_q = k_q * (k_q + 1) // 2
        num_params_r = k_r * (k_r + 1) // 2
        param_names_q    = [f'param_L_{i}' for i in range(num_params_q)]
        param_names_r    = [f'param_M_{i}' for i in range(num_params_r)]
        all_param_names  = ['eps_t', 'eps_o'] + param_names_q + param_names_r
        x_0              = ([eps_t, eps_o]
                            + [eps_l] * num_params_q
                            + [eps_m] * num_params_r)
        decorator_kwargs = {'params_L_vec': 'param_L_', 'params_M_vec': 'param_M_'}
        model_config     = {
            'k_q': k_q, 'k_r': k_r,
            'num_blocks_q': k_q, 'num_blocks_r': k_r,
        }

    elif noise_model == 'adaptive_bws':
        k_q = k_r = k_val
        num_params_q = k_q * (k_q + 1) // 2
        num_params_r = k_r * (k_r + 1) // 2
        param_names_q    = [f'init_L_{i}' for i in range(num_params_q)]
        param_names_r    = [f'init_M_{i}' for i in range(num_params_r)]
        all_param_names  = ['eps_t', 'eps_o'] + param_names_q + param_names_r
        x_0              = ([eps_t, eps_o]
                            + [eps_l] * num_params_q
                            + [eps_m] * num_params_r)
        decorator_kwargs = {'params_L_vec': 'init_L_', 'params_M_vec': 'init_M_'}
        model_config     = {
            'k_q': k_q, 'k_r': k_r,
            'num_blocks_q': k_q, 'num_blocks_r': k_r,
            'alpha': alpha, 'beta': beta, 'w': window_size,
        }

    elif noise_model == 'dd_aekf':
        all_param_names  = ['eps_t', 'eps_o', 'eps_q_dd', 'eps_r_dd']
        x_0              = [eps_t, eps_o, eps_q, eps_q]
        decorator_kwargs = {}
        model_config     = {}

    else:
        raise ValueError(f"Unknown noise_model: {noise_model}")

    model_config['kernel_type'] = 'rbf'

    # ── Build OELTO experiment ─────────────────────────────────────────────────
    experiment = OELTO(
        train_input=train_input,
        validation_input=val_input,
        validation_groundtruth=val_gt,
        test_input=final_test_input,
        test_groundtruth=final_test_gt,
        noise_model=noise_model,
        model_config=model_config,
    )

    # ── Total wall-clock start ────────────────────────────────────────────────
    t_start = time.time()

    # Step 1: operator training
    experiment.train_operators(epochs, batch_size, window_size, window_size)

    # Step 2: CMA-ES optimisation
    @parameter_naming(all_param_names)
    @parameter_arrays(**decorator_kwargs)
    @parameter_transform(np.exp)
    @exception_catcher(np.linalg.LinAlgError, 1e6)
    def objective(**kwargs):
        return experiment.validation(**kwargs)

    cma_opt = cma.CMAEvolutionStrategy(x_0, 0.5, {'verbose': -9, 'verb_disp': 0})
    cma_opt.optimize(objective_fct=objective, iterations=cma_iters, verb_disp=0)

    # Step 3: final test
    test_mse, _ = experiment.test_evaluation()

    t_total = time.time() - t_start

    return test_mse, t_total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ──────────────────────────────────────────────────────────────────────────
    # USER-EDITABLE CONSTANTS  ← change these directly
    # ──────────────────────────────────────────────────────────────────────────
    DIM_LIST    = [5, 10, 20, 40]   # state dimensions to sweep
    T_STEPS     = 1000              # integration steps
    F_FORCING   = 8.0               # Lorenz-96 forcing
    OBS_NOISE   = 0.1               # Gaussian obs noise std
    DATA_SEED   = 42                # fixed — reproducible data
    MODEL_SEED  = None              # None = not fixed

    WINDOW_SIZE = 5
    CMA_ITERS   = 10
    EPOCHS      = 200
    BATCH_SIZE  = 50
    ALPHA       = 0.9               # AKF Q memory factor
    BETA        = 0.9               # AKF R memory factor
    # ──────────────────────────────────────────────────────────────────────────

    # ── CLI: only seed overrides and log dir (low-frequency) ─────────────────
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-seed',  type=int,   default=DATA_SEED)
    ap.add_argument('--model-seed', type=int,   default=MODEL_SEED)
    ap.add_argument('--obs-noise',  type=float, default=OBS_NOISE,
                    help='Gaussian observation noise std (default: 0.1)')
    ap.add_argument('--logdir',     type=str,   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lorenz_log'))
    ap.add_argument('--name',       type=str,   default=None,
                    help='Optional suffix appended to log filename, e.g. --name fixedk5')
    cli = ap.parse_args()

    DATA_SEED  = cli.data_seed
    MODEL_SEED = cli.model_seed
    OBS_NOISE  = cli.obs_noise

    # ── Log file setup ────────────────────────────────────────────────────────
    dim_str   = '-'.join(str(d) for d in DIM_LIST)
    noise_str = str(OBS_NOISE).replace('.', 'p')   # 0.1 → 0p1
    base      = (f"lorenz96_D{dim_str}_T{T_STEPS}"
                 f"_F{int(F_FORCING)}_noise{noise_str}_dseed{DATA_SEED}")
    log_name  = f"{base}_{cli.name}.log" if cli.name else f"{base}.log"
    log_path  = os.path.join(cli.logdir, log_name)
    os.makedirs(cli.logdir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger()

    # ── Header ────────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("  ELTO-AKF  —  Lorenz-96 Scalability Experiment")
    log.info(f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("-" * 70)
    log.info(f"  Lorenz params : F={F_FORCING}, dt=0.01, T={T_STEPS}, "
             f"obs_noise={OBS_NOISE}")
    log.info(f"  Data seed     : {DATA_SEED}  (fixed — controls IC + obs noise)")
    log.info(f"  Model seed    : {MODEL_SEED}  "
             f"({'not fixed' if MODEL_SEED is None else 'fixed'})")
    log.info("-" * 70)
    log.info(f"  ELTO params   : epochs={EPOCHS}, batch={BATCH_SIZE}, "
             f"window={WINDOW_SIZE}, cma_iters={CMA_ITERS}")
    log.info(f"  AKF params    : alpha={ALPHA}, beta={BETA}, "
             f"k=max(2, D//8) per dimension")
    log.info("=" * 70)
    log.info(f"  {'D':>4}  {'Model':<10}  {'Test MSE':>14}  {'Total Time (s)':>16}")
    log.info("  " + "-" * 50)

    models_to_run = {
        # 'ELTO-KF':  'static',
        # 'ELTO-SB':  'fixed_bws',
        # 'ELTO-AKF': 'adaptive_bws',
        'DD-AEKF':  'dd_aekf',
        'SPKF-nUI': 'spkfnui',
    }

    results = []

    # ── Part 1 + 2: generate data then evaluate, one D at a time ─────────────
    with tempfile.TemporaryDirectory(prefix='lorenz96_') as tmp_dir:
        for D in DIM_LIST:
            k_val = max(2, D // 8)
            # SPKF-nUI hidden: scale with D but cap at 64
            spkf_hidden = min(64, max(16, D * 4))

            log.info(f"\n  [Generating]  D={D}  k={k_val}  spkf_hidden={spkf_hidden}")
            npz_path = save_lorenz96_npz(
                D, T_STEPS, F_FORCING, OBS_NOISE, DATA_SEED, tmp_dir)

            # Load npz splits once, reuse for SPKF-nUI
            raw              = np.load(npz_path)
            train_np         = raw['train_data'].astype(np.float32)
            test_np          = raw['test_obs'].astype(np.float32)
            gt_np            = raw['test_groundtruth'].astype(np.float32)
            if train_np.ndim == 1: train_np = train_np[:, None]
            if test_np.ndim  == 1: test_np  = test_np[:, None]
            if gt_np.ndim    == 1: gt_np    = gt_np[:, None]
            split            = len(test_np) // 2

            for model_name, noise_model_code in models_to_run.items():
                try:
                    if noise_model_code == 'spkfnui':
                        mse, t_total = run_spkfnui(
                            train_input = train_np,
                            val_input   = test_np[:split],
                            val_gt      = gt_np[:split],
                            test_input  = test_np[split:],
                            test_gt     = gt_np[split:],
                            gru_hidden  = spkf_hidden,
                            model_seed  = MODEL_SEED,
                        )
                    else:
                        mse, t_total = run_single_experiment(
                            dataset_path = npz_path,
                            noise_model  = noise_model_code,
                            window_size  = WINDOW_SIZE,
                            k_val        = k_val,
                            cma_iters    = CMA_ITERS,
                            epochs       = EPOCHS,
                            batch_size   = BATCH_SIZE,
                            alpha        = ALPHA,
                            beta         = BETA,
                            model_seed   = MODEL_SEED,
                        )
                    log.info(f"  {D:>4}  {model_name:<10}  "
                             f"{mse:>14.8f}  {t_total:>14.4f}s")
                    results.append({
                        'D': D, 'k': k_val,
                        'model': model_name,
                        'mse': mse,
                        'total_time_s': round(t_total, 4),
                    })

                except Exception as e:
                    log.info(f"  {D:>4}  {model_name:<10}  "
                             f"{'ERROR'!s:>14}  — {e}")
                    results.append({
                        'D': D, 'k': k_val,
                        'model': model_name,
                        'mse': None,
                        'total_time_s': None,
                    })

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  SUMMARY  (aggregated across dimensions)")
    log.info("  " + "-" * 50)
    log.info(f"  {'D':>4}  {'k':>4}  {'Model':<10}  "
             f"{'Test MSE':>14}  {'Total Time (s)':>16}")
    log.info("  " + "-" * 50)
    for r in results:
        mse_str  = f"{r['mse']:.8f}"  if r['mse']         is not None else 'ERROR'
        time_str = f"{r['total_time_s']:.4f}s" if r['total_time_s'] is not None else 'ERROR'
        log.info(f"  {r['D']:>4}  {r['k']:>4}  {r['model']:<10}  "
                 f"{mse_str:>14}  {time_str:>14}")
    log.info("=" * 70)
    log.info(f"\n  Log saved → {log_path}")


if __name__ == '__main__':
    main()
