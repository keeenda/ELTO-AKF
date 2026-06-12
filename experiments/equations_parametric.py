import time
import numpy as np
import cma
from matplotlib import pyplot as plt
from scipy import stats
from scipy.io import loadmat, savemat
import torch
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
from ssm_discovery.utils import read_2d_data
from model.oELTO_eval import OELTO
from model.utils import parameter_naming, parameter_transform, exception_catcher, parameter_arrays

# ================================================================================
# DATASET loading and settings

# equation_name = 'burgers_parametric'
# equation_name = 'burgers_shock'
equation_name = 'burgers'
noise_level = 50
index = 1
iterations = 100

data_dir = os.path.join(_REPO_ROOT, "ssm_discovery", "datasets")
output_dir = os.path.join(_REPO_ROOT, "ssm_discovery", "outputs")
os.makedirs(output_dir, exist_ok=True)
# noisy data
file_path = os.path.join(data_dir, f"{equation_name}_noisy{noise_level}.mat")
# file_path = os.path.join(data_dir, f"{equation_name}_nonstationary.mat")
# file_path = os.path.join(data_dir, f"{equation_name}_noisy{noise_level}.mat")
# clean data
clean_file_path = os.path.join(data_dir, f"{equation_name}.mat")

data = loadmat(file_path)

u, du, _, _ = read_2d_data(file_path)
# clean_u, _, _= read_2d_data(clean_file_path)
clean_u, _, _, _ = read_2d_data(clean_file_path)

train_input = du  # ytrain
validation_input = du  #
validation_groundtruth = u # yval 
test_input = u  # y test_obs
test_groundtruth = clean_u

print("="*120)
print(f"\n mse between clean and noisy: {np.mean((test_input - test_groundtruth) ** 2):.8f}")
print(f"mse between noisy and double noisy: {np.mean((train_input - test_input) ** 2):.8f}  ")

# hyper-param settings
epochs = 200
batch_size = 50
window_size = 20
d = window_size
N = train_input.shape[0] + 1 - 2 * window_size

# ================================================================================
# noise model and init

noise_model = 'static'
# noise_model = 'fixed_diag'
# noise_model = 'fixed_bws'
# noise_model = 'adaptive_bws'

eps_t = -11
eps_o = -8
eps_q = -10
eps_r = -8
eps_l = -3.5
eps_m = -5.5
diag_config = {'k_q': 5, 'k_r': 5}
bws_config = {'k_q': 5, 'k_r': 5}
adaptive_config = {'alpha': 0.9, 'beta': 0.9, 'w': window_size}

if noise_model == 'static':
    print("---  Static Mode ---")
    all_param_names = ['eps_t', 'eps_o', 'eps_q']
    x_0 = [eps_t, eps_o, eps_q]
    decorator_kwargs = {}
    model_config = {}

elif noise_model == 'fixed_diag':
    print("---  Fixed Diagonal Mode ---")
    model_config = diag_config 
    k_q = model_config.get('k_q')
    k_r = model_config.get('k_r')
    
    param_names_q = [f'eq_{i}' for i in range(k_q)]
    param_names_r = [f'er_{i}' for i in range(k_r)]
    all_param_names = ['eps_t', 'eps_o'] + param_names_q + param_names_r

    x0_q = [eps_q] * k_q
    x0_r = [eps_r] * k_r
    x_0 = [eps_t, eps_o] + x0_q + x0_r
    
    decorator_kwargs = {'eps_q_vec_k': 'eq_', 'eps_r_vec_k': 'er_'}

elif noise_model == 'fixed_bws':
    print("---  Fixed BWS Mode ---")
    model_config = bws_config # <--- 使用预定义的字典
    k_q = model_config.get('k_q', 2)
    k_r = model_config.get('k_r', 2)

    # 计算Q和R所需的底层因子参数数量
    num_params_q = k_q * (k_q + 1) // 2
    num_params_r = k_r * (k_r + 1) // 2

    param_names_q = [f'param_L_{i}' for i in range(num_params_q)]
    param_names_r = [f'param_M_{i}' for i in range(num_params_r)]
    all_param_names = ['eps_t', 'eps_o'] + param_names_q + param_names_r

    # 通过重复同一个初始值来构建初始猜测向量
    x0_q = [eps_l] * num_params_q
    x0_r = [eps_m] * num_params_r
    x_0 = [eps_t, eps_o] + x0_q + x0_r

    # 装饰器需要接收完整的参数向量
    decorator_kwargs = {'params_L_vec': 'param_L_', 'params_M_vec': 'param_M_'}

elif noise_model == 'adaptive_bws':
    print("---   Adaptive BWS Mode ---")
    # 合并 BWS 结构配置和 adaptive 平滑因子配置
    model_config = {**bws_config, **adaptive_config} # <--- 使用预定义的字典
    k_q = model_config.get('k_q', 2)
    k_r = model_config.get('k_r', 2)
    
    num_params_q = k_q * (k_q + 1) // 2
    num_params_r = k_r * (k_r + 1) // 2

    param_names_q = [f'init_L_{i}' for i in range(num_params_q)]
    param_names_r = [f'init_M_{i}' for i in range(num_params_r)]
    all_param_names = ['eps_t', 'eps_o'] + param_names_q + param_names_r

    x0_q = [eps_l] * num_params_q
    x0_r = [eps_m] * num_params_r
    x_0 = [eps_t, eps_o] + x0_q + x0_r

    decorator_kwargs = {'params_L_vec': 'init_L_', 'params_M_vec': 'init_M_'}

# ================================================================================
# EXPERIMENT settings (保持不变)

experiment = OELTO(
    train_input=train_input,
    validation_input=validation_input,
    validation_groundtruth=validation_groundtruth,
    test_input=test_input,
    test_groundtruth=test_groundtruth,
    noise_model=noise_model,
    model_config=model_config
)

# ================================================================================
# TRAIN

experiment.train_operators(epochs, batch_size, window_size, d)

# ================================================================================
# VALID via cma

# ================================================================================
# DECORATE 
@parameter_naming(all_param_names)
@parameter_arrays(**decorator_kwargs) 
@parameter_transform(np.exp)
@exception_catcher(np.linalg.LinAlgError, 1e6)

def objective_for_cma(**kwargs):
    return experiment.validation(**kwargs)

# ================================================================================
# CMA-ES 
# print(f"CMA-ES starting with initial parameters x_0: {x_0}")
cma_opt = cma.CMAEvolutionStrategy(x_0, 0.5)
cma_opt.optimize(objective_fct=objective_for_cma, iterations=iterations, verb_disp=1)
# print(f"\n Best valid loss:{experiment.best_validation_loss:.8f}")

# ================================================================================
# TEST and SAVING RESULT
print("="*60)
# print("Opt Done!")
print(f"train_loss: {experiment.best_validation_loss:.8f}")

# test
final_loss_on_test, final_mu_on_test = experiment.test_evaluation()
print(f"test_loss: {final_loss_on_test:.8f}")
# print(f"final_mu: {final_mu_on_test.shape}")

# save file
save_filename = f"{equation_name}_noisy{noise_level}_{noise_model}_{index}.mat"
# if 'k_q' in model_config:
#     save_filename = f"filtered_result_{noise_model}_kq{model_config['k_q']}_kr{model_config['k_r']}.mat"
save_path = os.path.join(output_dir, save_filename)
# 保存mu结果
savemat(save_path , {'mu': final_mu_on_test})
print(f"save file at {output_dir}, named : {save_filename}")

# (可选) 保存最佳参数
best_params_dict = experiment.best_params_on_validation
print(f"找到的最佳参数是: {best_params_dict}")
# np.save(f"best_params_{noise_model}.npy", best_params_dict)