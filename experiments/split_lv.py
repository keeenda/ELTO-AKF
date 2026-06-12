"""
lv_elto_eval.py
===============
Real-world Lotka-Volterra (predator-prey) filtering experiment
for ELTO-KF / ELTO-SB / ELTO-AKF.

Data layout (4 .npy files, shape (2, 200)):
  lv_100.npy            — clean groundtruth
  lv_100_noise1.npy     — low  noise observations
  lv_100_noise10.npy    — mid  noise observations
  lv_100_noise100.npy   — high noise observations

Since there is no separate groundtruth for train, we use:
  train  = noisy[:T//2]   (self-supervised, proxy gt = clean[:T//2])
  val    = noisy[T//2:]   + REAL clean groundtruth[T//2:]
  test   = val  (merged val+test — only 200 steps total, no budget to split)

MSE is computed against the REAL clean groundtruth on the test set.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER-EDITABLE CONSTANTS  (top of main)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DATA_DIR      = '.'          # directory containing the .npy files
  WINDOW_SIZES  = [3, 5]       # both will be run and logged separately
  CMA_ITERS     = 10
  EPOCHS        = 200
  BATCH_SIZE    = 50
  K_VAL         = 2            # SB block count (small, data is only 200 steps)
  ALPHA         = 0.9
  BETA          = 0.9
  MODEL_SEED    = None

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python lv_elto_eval.py
  python lv_elto_eval.py --data-dir /path/to/npy --model-seed 123
  python lv_elto_eval.py --name trial1
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import cma
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.oELTO_eval import OELTO
from model.utils import (parameter_naming, parameter_transform,
                         exception_catcher, parameter_arrays)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

# Noise level label → noisy filename stem
NOISE_FILES = {
    'noise1':   'lv_100_noise1.npy',
    'noise10':  'lv_100_noise10.npy',
    'noise100': 'lv_100_noise100.npy',
}
CLEAN_FILE = 'lv_100.npy'


def load_lv_splits(data_dir, noise_key, species_idx):
    """
    Double-denoising setup (mirrors equations_parametric.py):
      train_input            = noisy  (200,1)  ← double noisy, for training
      validation_input       = clean  (200,1)  ← single noisy (cleaner obs)
      validation_groundtruth = clean  (200,1)  ← same, no true gt available
      test_input             = clean  (200,1)  ← input for final denoising
      test_groundtruth       = clean  (200,1)  ← placeholder, MSE not reported

    species_idx: 0 or 1  — each species treated as independent 1D experiment
    """
    clean_path = os.path.join(data_dir, CLEAN_FILE)
    noisy_path = os.path.join(data_dir, NOISE_FILES[noise_key])

    clean = np.load(clean_path).T.astype(np.float32)   # (200, 2)
    noisy = np.load(noisy_path).T.astype(np.float32)   # (200, 2)

    # Extract single species → (200, 1)
    clean_sp = clean[:, species_idx:species_idx+1]
    noisy_sp = noisy[:, species_idx:species_idx+1]

    # Fit scaler on noisy (double noisy) portion
    scaler = StandardScaler()
    scaler.fit(noisy_sp)
    noisy_scaled = scaler.transform(noisy_sp).astype(np.float32)
    clean_scaled = scaler.transform(clean_sp).astype(np.float32)

    return {
        'train_input':            noisy_scaled,   # double noisy → train
        'validation_input':       clean_scaled,   # clean (single noisy) → val
        'validation_groundtruth': clean_scaled,   # placeholder
        'test_input':             clean_scaled,   # clean → test input
        'test_groundtruth':       clean_scaled,   # placeholder, MSE not used
    }


# ══════════════════════════════════════════════════════════════════════════════
# Single experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_single_experiment(splits, noise_model,
                          window_size=3, k_val=2,
                          cma_iters=10, epochs=200, batch_size=50,
                          alpha=0.9, beta=0.9,
                          model_seed=None):
    """
    Run one (noise_level × model × window_size) combination.
    Returns (test_mse, total_wall_time_s).
    """
    if model_seed is not None:
        np.random.seed(model_seed)

    eps_t, eps_o, eps_q, eps_l, eps_m = -5, -6, -10, -3, -3

    if noise_model == 'static':
        all_param_names  = ['eps_t', 'eps_o', 'eps_q']
        x_0              = [eps_t, eps_o, eps_q]
        decorator_kwargs = {}
        model_config     = {}

    elif noise_model == 'fixed_bws':
        k_q = k_r = k_val
        num_params_q     = k_q * (k_q + 1) // 2
        num_params_r     = k_r * (k_r + 1) // 2
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
        num_params_q     = k_q * (k_q + 1) // 2
        num_params_r     = k_r * (k_r + 1) // 2
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
    else:
        raise ValueError(f"Unknown noise_model: {noise_model}")

    model_config['kernel_type'] = 'rbf'

    experiment = OELTO(
        noise_model=noise_model,
        model_config=model_config,
        **splits,
    )

    t_start = time.time()

    experiment.train_operators(epochs, batch_size, window_size, window_size)

    @parameter_naming(all_param_names)
    @parameter_arrays(**decorator_kwargs)
    @parameter_transform(np.exp)
    @exception_catcher(np.linalg.LinAlgError, 1e6)
    def objective(**kwargs):
        return experiment.validation(**kwargs)

    cma_opt = cma.CMAEvolutionStrategy(x_0, 0.5, {'verbose': -9, 'verb_disp': 0})
    cma_opt.optimize(objective_fct=objective, iterations=cma_iters, verb_disp=0)

    test_mse, final_mu = experiment.test_evaluation()
    t_total = time.time() - t_start

    return test_mse, t_total, final_mu


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ──────────────────────────────────────────────────────────────────────────
    # USER-EDITABLE CONSTANTS
    # ──────────────────────────────────────────────────────────────────────────
    DATA_DIR     = os.path.join(
                       os.path.dirname(os.path.abspath(__file__)),
                       '..', 'data', 'lv'
                   )                  # <experiments>/../data/lv/
    WINDOW_SIZES = [5]       # both will be run
    CMA_ITERS    = 10
    EPOCHS       = 200
    BATCH_SIZE   = 50
    K_VAL        = 2            # SB block count
    ALPHA        = 0.9
    BETA         = 0.9
    MODEL_SEED   = None
    # ──────────────────────────────────────────────────────────────────────────

    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir',   type=str, default=DATA_DIR)
    ap.add_argument('--model-seed', type=int, default=MODEL_SEED)
    ap.add_argument('--logdir',     type=str,
                    default=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), 'lv_log'))
    ap.add_argument('--outdir',     type=str,
                    default=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), 'lv_output'))
    ap.add_argument('--name',       type=str, default=None,
                    help='Optional suffix for log filename')
    cli = ap.parse_args()

    DATA_DIR   = cli.data_dir
    MODEL_SEED = cli.model_seed

    # ── Log file ──────────────────────────────────────────────────────────────
    os.makedirs(cli.logdir, exist_ok=True)
    os.makedirs(cli.outdir, exist_ok=True)
    w_str    = '-'.join(str(w) for w in WINDOW_SIZES)
    base     = f"lv_w{w_str}_k{K_VAL}_cma{CMA_ITERS}"
    log_name = f"{base}_{cli.name}.log" if cli.name else f"{base}.log"
    log_path = os.path.join(cli.logdir, log_name)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger()

    models_to_run = {
        'ELTO-KF':  'static',
        'ELTO-SB':  'fixed_bws',
        # 'ELTO-AKF': 'adaptive_bws',
    }

    # ── Header ────────────────────────────────────────────────────────────────
    log.info("=" * 72)
    log.info("  ELTO-AKF  —  Lotka-Volterra Real-World Filtering Experiment")
    log.info(f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("-" * 72)
    log.info(f"  Data       : {DATA_DIR}/{CLEAN_FILE}  (groundtruth)")
    log.info(f"  Noise lvls : {list(NOISE_FILES.keys())}")
    log.info(f"  Split      : train=50%, test=50% (val merged with test)")
    log.info("-" * 72)
    log.info(f"  ELTO params: epochs={EPOCHS}, batch={BATCH_SIZE}, "
             f"cma_iters={CMA_ITERS}, k={K_VAL}")
    log.info(f"  AKF params : alpha={ALPHA}, beta={BETA}")
    log.info(f"  Model seed : {MODEL_SEED} "
             f"({'not fixed' if MODEL_SEED is None else 'fixed'})")
    log.info("=" * 72)

    results = []

    for window_size in WINDOW_SIZES:
        log.info(f"\n{'─' * 72}")
        log.info(f"  WINDOW SIZE = {window_size}")
        log.info(f"{'─' * 72}")
        log.info(f"  {'Species':<8}  {'Noise':<12}  {'Model':<10}  "
                 f"{'Total Time (s)':>16}")
        log.info("  " + "-" * 56)

        for sp_idx in range(2):                      # sp0 = species 0, sp1 = species 1
            for noise_key in NOISE_FILES:
                try:
                    splits = load_lv_splits(DATA_DIR, noise_key, sp_idx)
                except Exception as e:
                    log.info(f"  sp{sp_idx}  {noise_key:<12}  LOAD ERROR  — {e}")
                    continue

                for model_name, noise_model_code in models_to_run.items():
                    try:
                        mse, t_total, final_mu = run_single_experiment(
                            splits      = splits,
                            noise_model = noise_model_code,
                            window_size = window_size,
                            k_val       = K_VAL,
                            cma_iters   = CMA_ITERS,
                            epochs      = EPOCHS,
                            batch_size  = BATCH_SIZE,
                            alpha       = ALPHA,
                            beta        = BETA,
                            model_seed  = MODEL_SEED,
                        )
                        tag = 'kf' if noise_model_code == 'static' else 'akf'
                        out_name = f"lv_sp{sp_idx}_{noise_key}_{tag}.npy"
                        np.save(os.path.join(cli.outdir, out_name), final_mu)
                        log.info(f"  sp{sp_idx}  {noise_key:<12}  {model_name:<10}  "
                                 f"{t_total:>14.4f}s  → saved {out_name}")
                        results.append({
                            'window':  window_size,
                            'species': f'sp{sp_idx}',
                            'noise':   noise_key,
                            'model':   model_name,
                            'time_s':  round(t_total, 4),
                        })

                    except Exception as e:
                        log.info(f"  sp{sp_idx}  {noise_key:<12}  {model_name:<10}  "
                                 f"{'ERROR':>14}  — {e}")
                        results.append({
                            'window':  window_size,
                            'species': f'sp{sp_idx}',
                            'noise':   noise_key,
                            'model':   model_name,
                            'time_s':  None,
                        })

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info(f"\n{'=' * 72}")
    log.info("  SUMMARY  (MSE not reported — no true groundtruth)")
    log.info(f"  {'Window':<8}  {'Species':<8}  {'Noise':<12}  "
             f"{'Model':<12}  {'Total Time (s)':>14}")
    log.info("  " + "-" * 64)
    for r in results:
        time_str = f"{r['time_s']:.4f}s" if r['time_s'] is not None else 'ERROR'
        log.info(f"  {r['window']:<8}  {r['species']:<8}  {r['noise']:<12}  "
                 f"{r['model']:<12}  {time_str:>14}")
    log.info(f"{'=' * 72}")
    log.info(f"\n  Log saved → {log_path}")


if __name__ == '__main__':
    main()