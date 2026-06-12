# import numpy as np
# from torch.linalg import inv
# # from .deep_kernel_demo import Deep_Kernel, _rbf_kernel
# # from .deep_kernel import Deep_Kernel, _rbf_kernel, _linear_kernel, _poly_kernel
# from .deep_kernel import _rbf_kernel, _linear_kernel, _poly_kernel, _matern_kernel, _laplacian_kernel
# import torch
# import math

# class OAKF:
#     def __init__(self, trained_X, train_obs, window_size, d, device="cuda", 
#                  noise_model='static', model_config=None, is_deep_kernel=False):
#         super().__init__()

#         self.device = device
#         self.is_deep_kernel = is_deep_kernel
#         self.noise_model = noise_model
#         self.model_config = model_config if model_config is not None else {}

#         self.n_train = train_obs.shape[0]
#         self.h = window_size
#         self.N = self.n_train + 1 - 2 * self.h
#         self.X = trained_X
#         self.y = torch.tensor(train_obs, dtype=torch.float32).to(self.device)
#         if len(self.y.shape) == 1:
#             self.y = self.y.reshape(-1, 1)
#         if len(self.y.shape) == 1:
#             self.preimage_states = self.y[self.h:self.h + self.N]
#             self.preimage_states = self.preimage_states.reshape(-1, 1)
#         else:
#             self.preimage_states = self.y[self.h:self.h + self.N, :]
#         self.d = d
#         self.n_x0 = 200
#         self.alpha = 1
#         # self.rbf_kernel = _rbf_kernel
#         # self.deep_kernel = Deep_Kernel()
#         # self.kernel_x = self.rbf_kernel
#         # if self.is_deep_kernel == False:
#         #     self.kernel_y = self.rbf_kernel
#         # else:
#         #     with torch.no_grad():
#         #         self.kernel_y = self.deep_kernel
#         #         _, self.encoded_y = self.kernel_y(self.preimage_states)

#         # self.deep_kernel = Deep_Kernel()
#         # 1. 动态选择核心 kernel 函数
#         k_type = self.model_config.get('kernel_type', 'rbf')
#         if k_type == 'rbf':
#             chosen_kernel = _rbf_kernel
#         elif k_type == 'linear':
#             chosen_kernel = _linear_kernel
#         elif k_type == 'poly':
#             chosen_kernel = _poly_kernel
#         elif k_type == 'matern':      # <--- 新增
#             chosen_kernel = _matern_kernel
#         elif k_type == 'laplacian':   # <--- 新增
#             chosen_kernel = _laplacian_kernel
#         else:
#             raise ValueError(f"Unsupported kernel_type: {k_type}")

#         # 2. 赋值给 kernel_x 和 kernel_y
#         self.kernel_x = chosen_kernel
#         self.kernel_y = chosen_kernel
#         # if self.is_deep_kernel == False:
#         #     self.kernel_y = chosen_kernel
#         # else:
#         #     # self.deep_kernel = Deep_Kernel()
#         #     with torch.no_grad():
#         #         self.kernel_y = self.deep_kernel
#         #         _, self.encoded_y = self.kernel_y(self.preimage_states)

#         self.gram_X = None
#         self.gram_Y = None
#         self.EOO = None
#         self.ELTO = None
#         self.trans_mat0 = None
#         self.inn_res = None
#         self.GO = None
#         self.RG = None
#         self.YO = None
#         self.gram_K0 = None

#         # initialization
#         self.Q = None
#         self.R = None

#         if self.noise_model == 'fixed_diag':
#             self.params_q_diag = torch.empty(self.N - 1, device=self.device)
#             self.params_r_diag = torch.empty(self.N, device=self.device)

#         elif self.noise_model == 'fixed_bws' or self.noise_model == 'adaptive_bws' or self.noise_model == 'akf_ablation':
#             self.num_blocks_q = self.model_config.get('num_blocks_q', 2)
#             self.num_blocks_r = self.model_config.get('num_blocks_r', 2)

#             total_params_q = self.num_blocks_q * (self.num_blocks_q + 1) // 2
#             self.params_L = torch.empty(total_params_q, device=self.device)
#             total_params_r = self.num_blocks_r * (self.num_blocks_r + 1) // 2
#             self.params_M = torch.empty(total_params_r, device=self.device)

#         elif self.noise_model == 'ca_aekf':
#             dt = 1.0  # 假设采样间隔为 1
#             self.x_ca = torch.zeros((6, 1), device=self.device)
#             self.P_ca = torch.eye(6, device=self.device)
#             self.F_ca = torch.tensor([
#                 [1.0, 0.0, dt, 0.0, 0.5*dt**2, 0.0],
#                 [0.0, 1.0, 0.0, dt, 0.0, 0.5*dt**2],
#                 [0.0, 0.0, 1.0, 0.0, dt, 0.0],
#                 [0.0, 0.0, 0.0, 1.0, 0.0, dt],
#                 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
#             ], device=self.device)
            
#             self.H_ca = torch.tensor([
#                 [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
#             ], device=self.device)

#         elif self.noise_model == 'cv_aekf':
#             dt = 1.0  
#             self.x_cv = torch.zeros((4, 1), device=self.device)
#             self.P_cv = torch.eye(4, device=self.device)
            
#             # CV 模型：只有位置和速度，没有加速度项
#             self.F_cv = torch.tensor([
#                 [1.0, 0.0, dt, 0.0],
#                 [0.0, 1.0, 0.0, dt ],
#                 [0.0, 0.0, 1.0, 0.0],
#                 [0.0, 0.0, 0.0, 1.0]
#             ], device=self.device)
            
#             self.H_cv = torch.tensor([
#                 [1.0, 0.0, 0.0, 0.0],
#                 [0.0, 1.0, 0.0, 0.0]
#             ], device=self.device)

#         self.model_learned = False

#     def _construct_bws_matrix(self, factor_params_vec, total_dim, num_blocks):
#         """
#         Constructs a BWS matrix from a flat vector of its factor's scalar parameters.
#         """
#         # 1. 自动计算分块维度
#         block_dims = [total_dim // num_blocks + 1] * (total_dim % num_blocks) + \
#                     [total_dim // num_blocks] * (num_blocks - (total_dim % num_blocks))
        
#         # 2. 从参数向量构建因子矩阵 (factor_matrix)
#         factor_matrix = torch.zeros(total_dim, total_dim, device=self.device)
        
#         param_pointer = 0
#         row_start = 0
#         for i in range(num_blocks):
#             col_start = 0
#             for j in range(i + 1):
#                 p_ij = factor_params_vec[param_pointer]
#                 param_pointer += 1
                
#                 dim_i, dim_j = block_dims[i], block_dims[j]
#                 scalar_block = p_ij * torch.eye(dim_i, dim_j, device=self.device)
                
#                 factor_matrix[row_start:row_start+dim_i, col_start:col_start+dim_j] = scalar_block
#                 col_start += dim_j
#             row_start += dim_i
        
#         # 3. 计算 A = L * L^T 得到最终的BWS矩阵
#         bws_matrix = factor_matrix @ factor_matrix.T
#         return bws_matrix

#     def _decompose_bws_matrix(self, bws_matrix, num_blocks):
#         """
#         Performs a structured Cholesky decomposition on a BWS matrix to find its factor.
#         """
#         total_dim = bws_matrix.shape[0]

#         # 1. 提取 k x k 的标量系数矩阵
#         block_dims = [total_dim // num_blocks + 1] * (total_dim % num_blocks) + \
#                     [total_dim // num_blocks] * (num_blocks - (total_dim % num_blocks))
        
#         scalar_coeff_matrix = torch.zeros(num_blocks, num_blocks, device=self.device)
#         end_indices = torch.cumsum(torch.tensor(block_dims), dim=0)
#         start_indices = torch.cat((torch.tensor([0]), end_indices[:-1]))

#         for i in range(num_blocks):
#             for j in range(i + 1):
#                 scalar_val = bws_matrix[start_indices[i], start_indices[j]]
#                 scalar_coeff_matrix[i, j] = scalar_val
#                 if i != j:
#                     scalar_coeff_matrix[j, i] = scalar_val

#         # 2. 对系数矩阵进行稳健的Cholesky分解
#         try:
#             # --- 乐观尝试：直接进行Cholesky分解 ---
#             # 如果 scalar_coeff_matrix 已经是严格正定的，这一步会直接成功，效率最高。
#             factor_coeff_matrix = torch.linalg.cholesky(scalar_coeff_matrix)
            
#         except torch.linalg.LinAlgError:
#             # --- 修复流程：仅在直接分解失败时，才启动特征值裁剪 ---
#             # 这种情况意味着矩阵是半正定的（有零特征值）或不定的（有负特征值）。
#             # print("Warning: Direct Cholesky failed. Applying robust decomposition via eigenvalue clipping.")
            
#             try:
#                 # a. 对对称矩阵执行特征分解
#                 eigenvalues, eigenvectors = torch.linalg.eigh(scalar_coeff_matrix)
                
#                 # b. 裁剪特征值，确保它们至少为一个很小的正数
#                 min_eigenvalue = 1e-6
#                 clipped_eigenvalues = torch.clamp(eigenvalues, min=min_eigenvalue)
                
#                 # c. 重构一个保证严格正定的矩阵
#                 a_positive_definite = eigenvectors @ torch.diag(clipped_eigenvalues) @ eigenvectors.T
                
#                 # d. 对修复后的矩阵进行分解
#                 factor_coeff_matrix = torch.linalg.cholesky(a_positive_definite)

#             except torch.linalg.LinAlgError as e_final:
#                 # 如果修复后仍然失败（极罕见），打印致命错误并返回安全的默认值
#                 print(f"FATAL: Cholesky分解在特征值裁剪后仍然失败: {e_final}")
#                 factor_coeff_matrix = torch.eye(num_blocks, device=self.device)

#         # 3. 返回扁平化的因子参数向量 (逻辑不变)
#         indices = torch.tril_indices(row=num_blocks, col=num_blocks, offset=0)
#         flat_factor_params = factor_coeff_matrix[indices[0], indices[1]]
#         return flat_factor_params

#         # 2. 对系数矩阵进行Cholesky分解
#         # try:
#         #     factor_coeff_matrix = torch.linalg.cholesky(scalar_coeff_matrix)
#         # except torch.linalg.LinAlgError:
#         #     print("Warning: Cholesky decomposition failed on coefficient matrix. Adding regularization.")
#         #     a_reg = scalar_coeff_matrix + 1e-3 * torch.eye(num_blocks, device=self.device)
#         #     factor_coeff_matrix = torch.linalg.cholesky(a_reg)

#         # # 3. 用分解出的因子系数，重构完整的因子矩阵
#         # reconstructed_factor = torch.zeros_like(bws_matrix)
#         # row_start = 0
#         # for i in range(num_blocks):
#         #     col_start = 0
#         #     for j in range(i + 1):
#         #         p_ij = factor_coeff_matrix[i, j] # 从分解出的系数矩阵中取值
#         #         dim_i, dim_j = block_dims[i], block_dims[j]
#         #         scalar_block = p_ij * torch.eye(dim_i, dim_j, device=self.device)
#         #         reconstructed_factor[row_start:row_start+dim_i, col_start:col_start+dim_j] = scalar_block
#         #         col_start += dim_j
#         #     row_start += dim_i
            
#         # return reconstructed_factor

#     def learn_model(self, **kwargs):
#         """
#         Constructs noise matrices Q and R based on the current noise_model and
#         parameters from the optimizer, then learns the system operators ELTO and EOO.
#         """
#         # --- 1. 从kwargs中提取通用的超参数 ---
#         eps_t = kwargs.get('eps_t', np.exp(-6))
#         eps_o = kwargs.get('eps_o', np.exp(-10))

#         # --- 2. 根据 noise_model 构造 self.Q 和 self.R ---
#         if self.noise_model == 'static':
#             # static模式下，eps_q也是一个独立的参数
#             eps_q = kwargs['eps_q']
#             self.Q = 1e-5 * torch.eye(self.N - 1, device=self.device) # 或者使用eps_q
#             self.R = self.N * eps_q * torch.eye(self.N, device=self.device)

#         elif self.noise_model == 'fixed_diag':
#             # 装饰器已经将 'eq_...' 参数合并为 'eps_q_vec_k'
#             params_q_k = kwargs['eps_q_vec_k']
#             params_r_k = kwargs['eps_r_vec_k']

#             # 将main脚本中构建对角向量的逻辑迁移到这里
#             N_trans = self.N - 1
#             k_q = len(params_q_k)
#             chunk_size_q = N_trans // k_q
#             repeats_q = [chunk_size_q] * k_q
#             if k_q > 0 and N_trans > 0:
#                 repeats_q[-1] += N_trans % k_q
#             diag_vector_q = torch.from_numpy(np.repeat(params_q_k, repeats_q)).float().to(self.device)

#             N_obs = self.N
#             k_r = len(params_r_k)
#             chunk_size_r = N_obs // k_r
#             repeats_r = [chunk_size_r] * k_r
#             if k_r > 0 and N_obs > 0:
#                 repeats_r[-1] += N_obs % k_r
#             diag_vector_r = torch.from_numpy(np.repeat(params_r_k, repeats_r)).float().to(self.device)

#             self.Q = torch.diag(diag_vector_q)
#             self.R = torch.diag(diag_vector_r)

#         elif self.noise_model == 'fixed_bws' or self.noise_model == 'adaptive_bws' or self.noise_model == 'akf_ablation':
            
#             # 装饰器已经将 'param_L_...' 合并为 'params_L_vec'
#             # 对于 'adaptive_bws' 和 'akf_ablation'，这对应的是 'init_L_...' / 'init_M_...'
#             params_L_vec = torch.from_numpy(kwargs['params_L_vec']).float().to(self.device)
#             params_M_vec = torch.from_numpy(kwargs['params_M_vec']).float().to(self.device)
            
#             # 调用辅助函数来构造Q和R (作为 Q_0 和 R_0)
#             self.Q = self._construct_bws_matrix(params_L_vec, self.N - 1, self.num_blocks_q)
#             self.R = self._construct_bws_matrix(params_M_vec, self.N, self.num_blocks_r)

#         elif self.noise_model == 'ca_aekf':
#             eps_q_ca = kwargs.get('eps_q_ca', 1e-5)
#             eps_r_ca = kwargs.get('eps_r_ca', 1e-5)
#             self.Q = eps_q_ca * torch.eye(6, device=self.device)
#             self.R = eps_r_ca * torch.eye(2, device=self.device)
            
#             # 初始化状态
#             self.x_ca = torch.zeros((6, 1), device=self.device)
#             self.P_ca = torch.eye(6, device=self.device)
            
#             self.model_learned = True
#             return None, None  # 不需要 ELTO 和 EOO

#         elif self.noise_model == 'cv_aekf':
#             eps_q_cv = kwargs.get('eps_q_cv', 1e-5)
#             eps_r_cv = kwargs.get('eps_r_cv', 1e-5)
#             self.Q = eps_q_cv * torch.eye(4, device=self.device)
#             self.R = eps_r_cv * torch.eye(2, device=self.device)
            
#             self.x_cv = torch.zeros((4, 1), device=self.device)
#             self.P_cv = torch.eye(4, device=self.device)
#             self.model_learned = True
#             return None, None

#         elif self.noise_model == 'dd_aekf':
#             eps_q_dd = kwargs.get('eps_q_dd', 1e-5)
#             eps_r_dd = kwargs.get('eps_r_dd', 1e-5)
            
#             # --- 动态获取当前数据集的观测维度 ---
#             obs_dim = self.y.shape[1] 
            
#             self.Q = eps_q_dd * torch.eye(obs_dim, device=self.device)
#             self.R = eps_r_dd * torch.eye(obs_dim, device=self.device)
            
#             # Data-Driven System Identification
#             Y_past = self.y[:-1, :].T   
#             Y_future = self.y[1:, :].T  
            
#             self.F_dd = Y_future @ torch.linalg.pinv(Y_past)
#             self.H_dd = torch.eye(obs_dim, device=self.device)
            
#             self.x_dd = self.y[0:1, :].T 
#             self.P_dd = torch.eye(obs_dim, device=self.device)
            
#             self.model_learned = True
#             return None, None

#         # --- 3. 计算 Gram 矩阵 (逻辑不变) ---
#         self.gram_X = self.kernel_x(self.X.T)
#         gram_X2 = self.gram_X[:, 1:self.N]
#         if not self.is_deep_kernel:
#             self.gram_Y = self.kernel_y(self.preimage_states)
#         else:
#             self.gram_Y, _ = self.kernel_y(self.preimage_states)
        
#         # --- 4. 计算核心算子 ELTO 和 EOO (使用统一的 self.Q, self.R) ---
#         # (这部分逻辑与您原来基本一致，只是变量名更新了)
#         self.EOO = inv(self.gram_X + self.N * eps_o * torch.eye(self.N).T.to(self.device)) @ gram_X2
#         self.ELTO = inv(self.gram_X[:self.N-1,:self.N-1] + (self.N-1)
#                         * eps_t * torch.eye(self.N-1).to(self.device)).T @ self.gram_X[:self.N-1,1:self.N]

#         # --- 5. 计算其他辅助矩阵 (逻辑不变) ---
#         self.GO = self.gram_Y @ self.EOO
#         if self.is_deep_kernel:
#             self.YO = self.encoded_y.T @ self.EOO
#         else:
#             self.YO = self.preimage_states.T @ self.EOO 

#         # initial embeddings (逻辑不变)
#         X0 = torch.randn(self.n_x0, self.d).to(self.device)
#         self.gram_K0 = self.kernel_x(self.X[:,:self.N-1].T, X0)
#         self.trans_mat0 = inv(self.gram_X[:self.N-1, :self.N-1] + (self.N-1)
#                               * eps_t * torch.eye(self.N-1).to(self.device)).T @ self.gram_K0

#         self.model_learned = True
#         return self.ELTO, self.EOO

#     def update_step(self, m, cov, yt):

#         if self.is_deep_kernel == False:
#             g_Y = self.kernel_y(self.preimage_states, yt)
#         else:
#             with torch.no_grad():
#                 _, encoded_preimage = self.kernel_y(self.preimage_states)
#                 _, encoded_input = self.kernel_y(yt)
#                 g_Y = _rbf_kernel(encoded_preimage, encoded_input)

#         # kernel Kalman gain operator
#         O_cov = self.EOO @ cov
#         G_denominator_T = O_cov @ self.GO.T + self.R
#         G = self.solve_or_pinv(G_denominator_T, O_cov).T

#         # Update mean and covariance
#         d = g_Y - self.GO @ m
#         m_posterior = m + G @ d
#         cov_posterior = cov - G @ self.GO @ cov

#         # Calculate measurement residual
#         v = g_Y - self.GO @ m_posterior

#         return m_posterior, cov_posterior, v, d, G

#     def predict_step(self, m, cov):
#         # predict step, aka transition update
#         m = self.ELTO @ m

#         if cov is not None:
#             # --- MODIFIED: The prediction logic is universal. ---
#             # It always uses self.Q, which has been correctly set in learn_model.
#             # No if/else branches are needed here.
#             cov = self.ELTO @ cov @ self.ELTO.T + self.Q
#             cov = 0.5 * (cov + cov.T)
        
#         # --- NEW: Placeholder for adaptive Q update ---
#         # Note: A per-step update for Q is usually done here, after prediction.
#         # However, our windowed approach will handle this in the filter loop.
#         # For now, we follow the structure from update_step.
#         if self.noise_model == 'adaptive_bws':
#             pass

#         return m, cov

#     def pre_image(self, m, cov):
#         mu = self.YO @ m
#         sig = self.YO @ cov @ self.YO.T
#         return mu.T, sig

#     def initial_transition(self):
#         # N = self.N
#         # n_x0 = self.n_x0

#         # trans_mat0 = (inv(self.gram_X[:self.N-1, :self.N-1] + (self.N-1) * self.eps_t * torch.eye(self.N-1)).T @
#         #               self.gram_K0)
#         # self.learn_model()

#         m_t0 = torch.ones((self.n_x0, 1)).to(self.device) / self.n_x0
#         S_t0 = torch.eye(self.n_x0).to(self.device)
#         m_t = self.trans_mat0 @ m_t0  # 1st transition
#         S_t = self.trans_mat0 @ S_t0 @ self.trans_mat0.T + self.Q
#         return m_t, S_t

#     def solve_or_pinv(self, A, B, reg_lambda=1e-4):
#         """
#         尝试求解线性方程 AX = B，如果 A 是奇异的或病态的，使用伪逆。
#         使用正则化来增强数值稳定性。
#         """
#         try:
#             # 尝试使用 torch.linalg.solve
#             X = torch.linalg.solve(A, B)
#         except torch._C._LinAlgError as e:
#             # 如果 solve 失败了，使用 pinv 伪逆来求解
#             if 'singular' in str(e):
#                 # print("Matrix is singular or ill-conditioned, adding regularization term.")
#                 try:
#                     # 使用正则化增强矩阵 A 的数值稳定性
#                     A_reg = A + reg_lambda * torch.eye(A.shape[0], device=A.device)
#                     X = torch.matmul(torch.linalg.pinv(A_reg), B)
#                 except torch._C._LinAlgError as pinv_e:
#                     print(f"pinv also failed with error: {pinv_e}")
#                     # 最后手动计算伪逆（可能仍然失败）
#                     try:
#                         U, S, V = torch.svd(A_reg)
#                         # 正则化奇异值，避免病态问题
#                         S_inv = torch.diag(1.0 / (S + reg_lambda))
#                         X = V @ S_inv @ U.T @ B
#                     except torch._C._LinAlgError as svd_e:
#                         print(f"SVD still failed with error: {svd_e}")
#                         raise
#             else:
#                 # 重新抛出异常如果是其他错误
#                 raise
#         return X

#     # def _project_to_bws(self, dense_target_matrix, num_blocks):
#     #     """
#     #     Projects a dense matrix onto the Block-wise Scalar (BWS) structure.

#     #     This function performs the following steps:
#     #     1.  Calculates the dimensions for each block automatically.
#     #     2.  Iterates through each block of the dense input matrix.
#     #     3.  For each block, it calculates a single scalar value that best
#     #         represents the block's diagonal entries (we use the mean).
#     #     4.  Constructs a new matrix where each block is a scalar matrix
#     #         (scalar_value * Identity), thus conforming to the BWS structure.

#     #     Args:
#     #         dense_target_matrix (torch.Tensor): The dense matrix to be projected (e.g., R_target).
#     #         num_blocks (int): The number of blocks (k) in the BWS structure.

#     #     Returns:
#     #         torch.Tensor: A new matrix that has the BWS structure.
#     #     """
#     #     total_dim = dense_target_matrix.shape[0]

#     #     # 1. 自动计算分块维度
#     #     if num_blocks > 0:
#     #         block_dims = [total_dim // num_blocks + 1] * (total_dim % num_blocks) + \
#     #                      [total_dim // num_blocks] * (num_blocks - (total_dim % num_blocks))
#     #     else:
#     #         return torch.zeros_like(dense_target_matrix) # Handle edge case

#     #     # 2. 创建一个空的输出矩阵
#     #     bws_matrix = torch.zeros_like(dense_target_matrix)
        
#     #     # 预计算块的起始索引
#     #     end_indices = torch.cumsum(torch.tensor(block_dims), dim=0)
#     #     start_indices = torch.cat((torch.tensor([0]), end_indices[:-1]))

#     #     # 3. 遍历所有块，提取信息并构建BWS矩阵
#     #     for i in range(num_blocks):
#     #         for j in range(num_blocks):
#     #             # 获取当前块的行列索引
#     #             row_start, row_end = start_indices[i], end_indices[i]
#     #             col_start, col_end = start_indices[j], end_indices[j]
                
#     #             # a. 从稠密目标矩阵中提取对应的块
#     #             target_block = dense_target_matrix[row_start:row_end, col_start:col_end]
                
#     #             # b. 提取该块的对角线元素
#     #             diag_elements = torch.diag(target_block)
                
#     #             # c. 计算一个标量来代表整个块。
#     #             #    取对角线元素的平均值是一个非常稳健的选择。
#     #             if len(diag_elements) > 0:
#     #                 scalar_value = torch.mean(diag_elements)
#     #             else:
#     #                 scalar_value = 0.0 # Handle empty blocks if they occur
                
#     #             # d. 创建标量矩阵块，并放入输出矩阵的正确位置
#     #             #    torch.eye能正确处理非方形块
#     #             dim_i, dim_j = block_dims[i], block_dims[j]
#     #             scalar_block = scalar_value * torch.eye(dim_i, dim_j, device=self.device)
#     #             bws_matrix[row_start:row_end, col_start:col_end] = scalar_block
        
#     #     # 4. 为了严格满足对称性，取结果和其转置的平均值
#     #     return 0.5 * (bws_matrix + bws_matrix.T)

#     def _project_to_bws(self, dense_target_matrix, num_blocks, min_eigenvalue=1e-6):
#         """
#         Projects a dense matrix onto the Scalar-Block (SB) structure and strictly enforces
#         Positive Definiteness via Spectral Rectification.

#         This function implements the "Composite Projection Operator" described in the paper:
#         1. Structural Projection: Projects dense matrix to SB structure via trace averaging.
#         2. Spectral Rectification: Clamps eigenvalues to ensure SPD property.

#         Args:
#             dense_target_matrix (torch.Tensor): The dense matrix to be projected.
#             num_blocks (int): The number of blocks (k) in the SB structure.
#             min_eigenvalue (float): The threshold xi for spectral clamping (default: 1e-6).

#         Returns:
#             torch.Tensor: A strictly SPD matrix with SB structure.
#         """
#         total_dim = dense_target_matrix.shape[0]

#         # 1. 自动计算分块维度
#         if num_blocks > 0:
#             block_dims = [total_dim // num_blocks + 1] * (total_dim % num_blocks) + \
#                          [total_dim // num_blocks] * (num_blocks - (total_dim % num_blocks))
#         else:
#             return torch.zeros_like(dense_target_matrix) # Handle edge case

#         # 2. 创建一个空的输出矩阵
#         bws_matrix = torch.zeros_like(dense_target_matrix)
        
#         # 预计算块的起始索引
#         end_indices = torch.cumsum(torch.tensor(block_dims), dim=0)
#         start_indices = torch.cat((torch.tensor([0]), end_indices[:-1]))

#         # 3. Structural Projection: 遍历所有块，强制执行 Scalar-Block 结构
#         for i in range(num_blocks):
#             for j in range(num_blocks):
#                 # 获取当前块的行列索引
#                 row_start, row_end = start_indices[i], end_indices[i]
#                 col_start, col_end = start_indices[j], end_indices[j]
                
#                 # a. 从稠密目标矩阵中提取对应的块
#                 target_block = dense_target_matrix[row_start:row_end, col_start:col_end]
                
#                 # b. 提取该块的对角线元素
#                 diag_elements = torch.diag(target_block)
                
#                 # c. 计算一个标量来代表整个块 (Trace Averaging)
#                 #    对应论文公式: theta_ij = tr(A_ij) / rank(A_ij)
#                 if len(diag_elements) > 0:
#                     scalar_value = torch.mean(diag_elements)
#                 else:
#                     scalar_value = 0.0
                
#                 # d. 创建标量矩阵块
#                 dim_i, dim_j = block_dims[i], block_dims[j]
#                 scalar_block = scalar_value * torch.eye(dim_i, dim_j, device=self.device)
#                 bws_matrix[row_start:row_end, col_start:col_end] = scalar_block
        
#         # 4. Symmetry Enforcement: 严格保证对称性
#         #    这是进行特征值分解的前提条件
#         bws_symmetric = 0.5 * (bws_matrix + bws_matrix.T)

#         # 5. Spectral Rectification: 谱修正 (The Reviewer Fix)
#         #    对应论文公式: A^S = V * max(Lambda, xi * I) * V^T
#         #    使用 eigh 是因为矩阵是对称的，比 eig 更快且数值更稳定
#         eigs, vecs = torch.linalg.eigh(bws_symmetric)
        
#         #    强制将小于阈值的特征值拉回到 min_eigenvalue
#         eigs_clamped = torch.clamp(eigs, min=min_eigenvalue)
        
#         #    重构矩阵
#         bws_spd = vecs @ torch.diag(eigs_clamped) @ vecs.T
        
#         return bws_spd

#     def filter(self, observations):
#         assert self.model_learned

#         if self.noise_model == 'dd_aekf':
#             T = observations.shape[0]
#             obs_dim = observations.shape[1] # <--- 动态获取维度
            
#             # 初始化预测输出，匹配动态维度
#             y_mu_pred = torch.zeros((T, obs_dim), device=self.device)
            
#             window_size = self.model_config.get('w', 20)
#             alpha = self.model_config.get('alpha', 0.99)
#             beta = self.model_config.get('beta', 0.99)
            
#             residual_history, innovation_history, gain_history = [], [], []
            
#             for t in range(T):
#                 y_t = observations[t:t+1, :].T
                
#                 x_minus = self.F_dd @ self.x_dd
#                 P_minus = self.F_dd @ self.P_dd @ self.F_dd.T + self.Q
                
#                 d_t = y_t - (self.H_dd @ x_minus)
#                 S = self.H_dd @ P_minus @ self.H_dd.T + self.R
                
#                 # 正则化矩阵维度也要动态化
#                 S_reg = S + 1e-8 * torch.eye(obs_dim, device=self.device)
                
#                 try:
#                     inv_S = torch.linalg.inv(S_reg)
#                 except torch._C._LinAlgError:
#                     y_mu_pred[t:, :] = 100.0  
#                     break
                    
#                 K_t = P_minus @ self.H_dd.T @ inv_S
                
#                 self.x_dd = x_minus + K_t @ d_t
                
#                 # 更新 P 矩阵
#                 self.P_dd = (torch.eye(obs_dim, device=self.device) - K_t @ self.H_dd) @ P_minus
                
#                 v_t = y_t - (self.H_dd @ self.x_dd)
#                 y_mu_pred[t, :] = (self.H_dd @ self.x_dd).squeeze()
                
#                 residual_history.append(v_t)
#                 innovation_history.append(d_t)
#                 gain_history.append(K_t)
                
#                 if (t + 1) % window_size == 0 and t > 0:
#                     sum_vvT = torch.sum(torch.stack([v @ v.T for v in residual_history]), dim=0)
#                     R_hat = (1/window_size) * sum_vvT + self.H_dd @ self.P_dd @ self.H_dd.T
#                     self.R = alpha * self.R + (1 - alpha) * R_hat
#                     self.R = 0.5 * (self.R + self.R.T) 
                    
#                     sum_KdKd_T = torch.sum(torch.stack([(K @ d) @ (K @ d).T for K, d in zip(gain_history, innovation_history)]), dim=0)
#                     Q_hat = (1/window_size) * sum_KdKd_T
#                     self.Q = beta * self.Q + (1 - beta) * Q_hat
#                     self.Q = 0.5 * (self.Q + self.Q.T)
                    
#                     residual_history.clear()
#                     innovation_history.clear()
#                     gain_history.clear()

#             # 最后返回的默认协方差形状也要匹配
#             return y_mu_pred, torch.zeros(T, obs_dim, obs_dim).to(self.device)

#         # ==========================================================
#         # 拦截：如果处于 CA-AEKF 模式，直接运行原始物理空间的滤波并返回
#         # ==========================================================
#         if self.noise_model == 'ca_aekf':
#             T = observations.shape[0]
#             y_mu_pred = torch.zeros((T, 2), device=self.device)
            
#             window_size = self.model_config.get('w', 20)
#             alpha = self.model_config.get('alpha', 0.99)
#             beta = self.model_config.get('beta', 0.99)
            
#             residual_history, innovation_history, gain_history = [], [], []
            
#             for t in range(T):
#                 y_t = observations[t:t+1, :].T  # (2, 1)
                
#                 # Predict
#                 x_minus = self.F_ca @ self.x_ca
#                 P_minus = self.F_ca @ self.P_ca @ self.F_ca.T + self.Q
                
#                 # Update
#                 d_t = y_t - (self.H_ca @ x_minus)
#                 S = self.H_ca @ P_minus @ self.H_ca.T + self.R
#                 S_reg = S + 1e-6 * torch.eye(2, device=self.device)
#                 K_t = P_minus @ self.H_ca.T @ torch.linalg.inv(S_reg)
                
#                 self.x_ca = x_minus + K_t @ d_t
#                 self.P_ca = (torch.eye(6, device=self.device) - K_t @ self.H_ca) @ P_minus
                
#                 v_t = y_t - (self.H_ca @ self.x_ca)
#                 y_mu_pred[t, :] = (self.H_ca @ self.x_ca).squeeze()
                
#                 # 自适应矩阵更新 (标准 AEKF 逻辑，无 Scalar-Block)
#                 residual_history.append(v_t)
#                 innovation_history.append(d_t)
#                 gain_history.append(K_t)
                
#                 if (t + 1) % window_size == 0 and t > 0:
#                     sum_vvT = torch.sum(torch.stack([v @ v.T for v in residual_history]), dim=0)
#                     R_hat = (1/window_size) * sum_vvT + self.H_ca @ self.P_ca @ self.H_ca.T
#                     self.R = alpha * self.R + (1 - alpha) * R_hat
#                     self.R = 0.5 * (self.R + self.R.T) + 1e-6 * torch.eye(2, device=self.device)
                    
#                     sum_KdKd_T = torch.sum(torch.stack([(K @ d) @ (K @ d).T for K, d in zip(gain_history, innovation_history)]), dim=0)
#                     Q_hat = (1/window_size) * sum_KdKd_T
#                     self.Q = beta * self.Q + (1 - beta) * Q_hat
#                     self.Q = 0.5 * (self.Q + self.Q.T) + 1e-6 * torch.eye(6, device=self.device)
                    
#                     residual_history.clear()
#                     innovation_history.clear()
#                     gain_history.clear()

#             # CA-AEKF 直接返回结果，不计算复杂的 RKHS 协方差
#             return y_mu_pred, torch.zeros(T, 2, 2).to(self.device)

#         # ==========================================================
#         # 拦截 2：CV-AEKF 模式 (用于展示模型失配和无结构化导致的协方差崩溃)
#         # ==========================================================
#         if self.noise_model == 'cv_aekf':
#             T = observations.shape[0]
#             y_mu_pred = torch.zeros((T, 2), device=self.device)
            
#             window_size = self.model_config.get('w', 20)
#             alpha = self.model_config.get('alpha', 0.99)
#             beta = self.model_config.get('beta', 0.99)
            
#             residual_history, innovation_history, gain_history = [], [], []
            
#             for t in range(T):
#                 y_t = observations[t:t+1, :].T
                
#                 # Predict
#                 x_minus = self.F_cv @ self.x_cv
#                 P_minus = self.F_cv @ self.P_cv @ self.F_cv.T + self.Q
                
#                 # Update
#                 d_t = y_t - (self.H_cv @ x_minus)
#                 S = self.H_cv @ P_minus @ self.H_cv.T + self.R
                
#                 # 给一个极小正则化以防最初几步直接死掉
#                 S_reg = S + 1e-8 * torch.eye(2, device=self.device)
                
#                 try:
#                     inv_S = torch.linalg.inv(S_reg)
#                 except torch._C._LinAlgError:
#                     # 如果矩阵失去正定性导致求逆失败，直接捕获并惩罚！
#                     y_mu_pred[t:, :] = 100.0  # 赋予极大误差
#                     break
                    
#                 K_t = P_minus @ self.H_cv.T @ inv_S
                
#                 self.x_cv = x_minus + K_t @ d_t
#                 self.P_cv = (torch.eye(4, device=self.device) - K_t @ self.H_cv) @ P_minus
                
#                 v_t = y_t - (self.H_cv @ self.x_cv)
#                 y_mu_pred[t, :] = (self.H_cv @ self.x_cv).squeeze()
                
#                 residual_history.append(v_t)
#                 innovation_history.append(d_t)
#                 gain_history.append(K_t)
                
#                 if (t + 1) % window_size == 0 and t > 0:
#                     sum_vvT = torch.sum(torch.stack([v @ v.T for v in residual_history]), dim=0)
#                     R_hat = (1/window_size) * sum_vvT + self.H_cv @ self.P_cv @ self.H_cv.T
#                     self.R = alpha * self.R + (1 - alpha) * R_hat
#                     # 故意不加强制正定性保护，暴露出标准 AEKF 在非平稳高噪下的崩溃问题
#                     self.R = 0.5 * (self.R + self.R.T) 
                    
#                     sum_KdKd_T = torch.sum(torch.stack([(K @ d) @ (K @ d).T for K, d in zip(gain_history, innovation_history)]), dim=0)
#                     Q_hat = (1/window_size) * sum_KdKd_T
#                     self.Q = beta * self.Q + (1 - beta) * Q_hat
#                     self.Q = 0.5 * (self.Q + self.Q.T)
                    
#                     residual_history.clear()
#                     innovation_history.clear()
#                     gain_history.clear()

#             return y_mu_pred, torch.zeros(T, 2, 2).to(self.device)

#         m_t, S_t = self.initial_transition()
#         if self.is_deep_kernel == False:
#             y_mu_pred = torch.zeros(observations.shape).to(self.device)
#             y_sigma_pred = torch.zeros(observations.shape[0], observations.shape[1], observations.shape[1]).to(self.device)
#         else:
#             y_mu_pred = torch.zeros(observations.shape[0], 200).to(self.device)
#             y_sigma_pred = torch.zeros(observations.shape[0], 200, 200).to(self.device)

#         # --- MODIFICATION: 为 'adaptive_bws' 和 'akf_ablation' 共同初始化 ---
#         if self.noise_model == 'adaptive_bws' or self.noise_model == 'akf_ablation':
#             # 存储窗口内的信息
#             residual_history = []
#             innovation_history = []
#             gain_history = []
            
#             # 从配置中获取窗口大小 (w, a, b)
#             # 对应论文 Algorithm 1 中的 w, α, β
#             window_size = self.model_config.get('w', 20)
#             alpha = self.model_config.get('alpha', 0.99)
#             beta = self.model_config.get('beta', 0.99)

#             if self.noise_model == 'akf_ablation':
#                 # 消融实验：确保 Q 和 R 是稠密矩阵
#                 # 假设 Q 和 R 已由 learn_model (例如 'fixed_bws') 初始化
#                 # 确保它们不是对角稀疏矩阵（如果从 fixed_diag 初始化）
#                 if hasattr(self.Q, 'is_sparse') and self.Q.is_sparse:
#                     self.Q = self.Q.to_dense()
#                 if hasattr(self.R, 'is_sparse') and self.R.is_sparse:
#                     self.R = self.R.to_dense()
#         # --- MODIFICATION END ---


#         for t in range(observations.shape[0]):

#             # innovation step / update step
#             try:
#                 m_t, S_t, v_t, d_t, K_t = self.update_step(m_t, S_t, observations[t:t+1, :])
#             except torch._C._LinAlgError:
#                 # print("yedman /n ")
#                 y_mu_pred = 100 * torch.ones(observations.shape).to(self.device)
#                 y_sigma_pred = 100 * torch.ones(observations.shape[0], observations.shape[1], observations.shape[1]).to(self.device)
#                 break
#             # ignore use_posterior_decoding
#             y_mu_pred[t, :], y_sigma_pred[t, :] = self.pre_image(m_t, S_t)

#             # predict step / transition update
#             m_t, S_t = self.predict_step(m_t, S_t)

#             # --- NEW: Ablation Study Logic ---
#             if self.noise_model == 'akf_ablation':
#                 # 1. 累积历史信息
#                 residual_history.append(v_t)
#                 innovation_history.append(d_t)
#                 gain_history.append(K_t)

#                 # 2. 窗口更新 (标准AEKF，无结构化)
#                 if (t + 1) % window_size == 0 and t > 0:
                    
#                     # --- R (观测噪声) 的直接更新 ---
#                     # i. 计算目标 R_k (对应论文 Eq. 24 中括号内的项) [cite: 191]
#                     sum_vvT = torch.sum(torch.stack([v @ v.T for v in residual_history]), dim=0)
#                     R_hat = (1/window_size) * sum_vvT + self.EOO @ S_t @ self.EOO.T
                    
#                     # ii. 直接平滑更新 R 矩阵 (无投影, 无分解)
#                     # 对应论文 Eq. 24: C_W,t = α*C_W,t-1 + (1-α)*(...) [cite: 191]
#                     self.R = alpha * self.R + (1 - alpha) * R_hat
                    
#                     # 强制对称性和正定性 (SPD)
#                     # 论文提到这是标准AEKF的难点 [cite: 193, 196]
#                     self.R = 0.5 * (self.R + self.R.T)
#                     self.R += 1e-6 * torch.eye(self.R.shape[0], device=self.device)

#                     # --- Q (过程噪声) 的直接更新 ---
#                     # i. 计算目标 Q_k (对应论文 Eq. 25 中括号内的项) [cite: 192]
#                     sum_KdKd_T = torch.sum(torch.stack([(K @ d) @ (K @ d).T 
#                                                        for K, d in zip(gain_history, innovation_history)]), dim=0)
#                     Q_hat = (1/window_size) * sum_KdKd_T
                    
#                     # ii. 直接平滑更新 Q 矩阵 (无投影, 无分解)
#                     # 对应论文 Eq. 25: C_V,t = β*C_V,t-1 + (1-β)*(...) [cite: 192]
#                     self.Q = beta * self.Q + (1 - beta) * Q_hat

#                     # 强制对称性和正定性 (SPD)
#                     self.Q = 0.5 * (self.Q + self.Q.T)
#                     self.Q += 1e-6 * torch.eye(self.Q.shape[0], device=self.device)

#                     # c. 为下一个窗口清空历史记录
#                     residual_history.clear()
#                     innovation_history.clear()
#                     gain_history.clear()

#             # adaptive (你的原始 'adaptive_bws' 逻辑)
#             if self.noise_model == 'adaptive_bws':
#                 # (这里的代码与你原来发给我的保持一致)
#                 residual_history.append(v_t)
#                 innovation_history.append(d_t)
#                 gain_history.append(K_t)
#                 if (t + 1) % window_size == 0 and t > 0:
#                     # --- R 和 M 的迭代更新 (结构化) ---
#                     # i. 计算目标 R_k
#                     sum_vvT = torch.sum(torch.stack([v @ v.T for v in residual_history]), dim=0)
#                     R_hat = (1/window_size) * sum_vvT + self.EOO @ S_t @ self.EOO.T
                    
#                     # ii. 投影并分解 (论文核心贡献) [cite: 194, 202]
#                     R_S_hat = self._project_to_bws(R_hat, self.num_blocks_r)
#                     M_hat = self._decompose_bws_matrix(R_S_hat, self.num_blocks_r)
                    
#                     # iii. 平滑更新 M 的底层参数
#                     self.params_M.data = alpha * self.params_M.data + (1 - alpha) * M_hat
                    
#                     # iv. 为下一个窗口重构 R 矩阵
#                     self.R = self._construct_bws_matrix(self.params_M, self.N, self.num_blocks_r)

#                     # --- Q 和 L 的迭代更新 (结构化) ---
#                     # i. 计算目标 Q_k
#                     sum_KdKd_T = torch.sum(torch.stack([(K @ d) @ (K @ d).T 
#                                                        for K, d in zip(gain_history, innovation_history)]), dim=0)
#                     Q_hat = (1/window_size) * sum_KdKd_T
                    
#                     # ii. 投影并分解 [cite: 194, 202]
#                     Q_S_hat = self._project_to_bws(Q_hat, self.num_blocks_q)
#                     L_hat = self._decompose_bws_matrix(Q_S_hat, self.num_blocks_q)
                    
#                     # iii. 平滑更新 L 的底层参数
#                     self.params_L.data = beta * self.params_L.data + (1 - beta) * L_hat
                    
#                     # iv. 为下一个窗口重构 Q 矩阵
#                     self.Q = self._construct_bws_matrix(self.params_L, self.N - 1, self.num_blocks_q)
                    
#                     # c. 为下一个窗口清空历史记录
#                     residual_history.clear()
#                     innovation_history.clear()
#                     gain_history.clear()

#         return y_mu_pred, y_sigma_pred

#         # def filter(self, observations):
#     #     assert self.model_learned
#     #     m_t, S_t = self.initial_transition()
#     #     if self.is_deep_kernel == False:
#     #         y_mu_pred = torch.zeros(observations.shape).to(self.device)
#     #         y_sigma_pred = torch.zeros(observations.shape[0], observations.shape[1], observations.shape[1]).to(self.device)
#     #     else:
#     #         y_mu_pred = torch.zeros(observations.shape[0], 200).to(self.device)
#     #         y_sigma_pred = torch.zeros(observations.shape[0], 200, 200).to(self.device)

#     #     if self.noise_model == 'adaptive_bws':
#     #         # 存储窗口内的信息
#     #         residual_history = []
#     #         innovation_history = []
#     #         gain_history = []
            
#     #         # 从配置中获取窗口大小 (w, a, b)
#     #         window_size = self.model_config.get('w', 20)
#     #         alpha = self.model_config.get('alpha', 0.99)
#     #         beta = self.model_config.get('beta', 0.99)

#     #     for t in range(observations.shape[0]):

#     #         # innovation step / update step
#     #         try:
#     #             m_t, S_t, v_t, d_t, K_t = self.update_step(m_t, S_t, observations[t:t+1, :])
#     #         except torch._C._LinAlgError:
#     #             # print("yedman /n ")
#     #             y_mu_pred = 100 * torch.ones(observations.shape).to(self.device)
#     #             y_sigma_pred = 100 * torch.ones(observations.shape[0], observations.shape[1], observations.shape[1]).to(self.device)
#     #             break
#     #         # ignore use_posterior_decoding
#     #         y_mu_pred[t, :], y_sigma_pred[t, :] = self.pre_image(m_t, S_t)

#     #         # predict step / transition update
#     #         m_t, S_t = self.predict_step(m_t, S_t)

#     #         # ablation
#     #         # if self.noise_model == 'akf_ablation':


#     #         # adaptive
#     #         if self.noise_model == 'adaptive_bws':
#     #             residual_history.append(v_t)
#     #             innovation_history.append(d_t)
#     #             gain_history.append(K_t)
#     #             if (t + 1) % window_size == 0 and t > 0:
#     #                 sum_vvT = torch.sum(torch.stack([v @ v.T for v in residual_history]), dim=0)
#     #                 # R_hat = (1/window_size) * sum_vvT + self.EOO @ S_t @ self.GO.T # 简化的 HPH^T
#     #                 R_hat = (1/window_size) * sum_vvT + self.EOO @ S_t @ self.EOO.T
                    
#     #                 # ii. 投影并分解
#     #                 R_S_hat = self._project_to_bws(R_hat, self.num_blocks_r)
#     #                 M_hat = self._decompose_bws_matrix(R_S_hat, self.num_blocks_r)
                    
#     #                 # iii. 平滑更新 M 的底层参数
#     #                 self.params_M.data = alpha * self.params_M.data + (1 - alpha) * M_hat
                    
#     #                 # iv. 为下一个窗口重构 R 矩阵
#     #                 self.R = self._construct_bws_matrix(self.params_M, self.N, self.num_blocks_r)

#     #                 # --- Q 和 L 的迭代更新 ---
#     #                 # i. 计算目标 Q_k
#     #                 sum_KdKd_T = torch.sum(torch.stack([(K @ d) @ (K @ d).T 
#     #                                                    for K, d in zip(gain_history, innovation_history)]), dim=0)
#     #                 Q_hat = (1/window_size) * sum_KdKd_T
                    
#     #                 # ii. 投影并分解
#     #                 Q_S_hat = self._project_to_bws(Q_hat, self.num_blocks_q)
#     #                 L_hat = self._decompose_bws_matrix(Q_S_hat, self.num_blocks_q)
                    
#     #                 # iii. 平滑更新 L 的底层参数
#     #                 self.params_L.data = beta * self.params_L.data + (1 - beta) * L_hat
                    
#     #                 # iv. 为下一个窗口重构 Q 矩阵
#     #                 self.Q = self._construct_bws_matrix(self.params_L, self.N - 1, self.num_blocks_q)
                    
#     #                 # c. 为下一个窗口清空历史记录
#     #                 residual_history.clear()
#     #                 innovation_history.clear()
#     #                 gain_history.clear()

#     #     return y_mu_pred, y_sigma_pred

import numpy as np
from torch.linalg import inv
# from .deep_kernel_demo import Deep_Kernel, _rbf_kernel
# from .deep_kernel import Deep_Kernel, _rbf_kernel, _linear_kernel, _poly_kernel
from .deep_kernel import _rbf_kernel, _linear_kernel, _poly_kernel, _matern_kernel, _laplacian_kernel
import torch
import math

class OAKF:
    """Online Adaptive Kalman Filter (OAKF) operating in RKHS latent space.

    Naming convention — code vs. paper:
      "BWS" (Block-Wise Scalar) in this file  ==  "SB" (Scalar-Block) in the paper.
      noise_model='fixed_bws'    ->  ELTO-SB  in paper  (fixed SB covariance, tuned by CMA-ES)
      noise_model='adaptive_bws' ->  ELTO-AKF in paper  (online SB covariance adaptation)
    """
    def __init__(self, trained_X, train_obs, window_size, d, device="cpu",
                 noise_model='static', model_config=None, is_deep_kernel=False):
        super().__init__()

        self.device = device
        self.is_deep_kernel = is_deep_kernel
        self.noise_model = noise_model
        self.model_config = model_config if model_config is not None else {}

        self.n_train = train_obs.shape[0]
        self.h = window_size
        self.N = self.n_train + 1 - 2 * self.h
        self.X = trained_X
        self.y = torch.tensor(train_obs, dtype=torch.float32).to(self.device)
        if len(self.y.shape) == 1:
            self.y = self.y.reshape(-1, 1)
        if len(self.y.shape) == 1:
            self.preimage_states = self.y[self.h:self.h + self.N]
            self.preimage_states = self.preimage_states.reshape(-1, 1)
        else:
            self.preimage_states = self.y[self.h:self.h + self.N, :]
        self.d = d
        self.n_x0 = 200
        self.alpha = 1
        # self.rbf_kernel = _rbf_kernel
        # self.deep_kernel = Deep_Kernel()
        # self.kernel_x = self.rbf_kernel
        # if self.is_deep_kernel == False:
        #     self.kernel_y = self.rbf_kernel
        # else:
        #     with torch.no_grad():
        #         self.kernel_y = self.deep_kernel
        #         _, self.encoded_y = self.kernel_y(self.preimage_states)

        # self.deep_kernel = Deep_Kernel()
        # Select kernel function (default: RBF); override via model_config['kernel_type']
        k_type = self.model_config.get('kernel_type', 'rbf')
        if k_type == 'rbf':
            chosen_kernel = _rbf_kernel
        elif k_type == 'linear':
            chosen_kernel = _linear_kernel
        elif k_type == 'poly':
            chosen_kernel = _poly_kernel
        elif k_type == 'matern':
            chosen_kernel = _matern_kernel
        elif k_type == 'laplacian':
            chosen_kernel = _laplacian_kernel
        else:
            raise ValueError(f"Unsupported kernel_type: {k_type}")

        self.kernel_x = chosen_kernel
        self.kernel_y = chosen_kernel
        # if self.is_deep_kernel == False:
        #     self.kernel_y = chosen_kernel
        # else:
        #     # self.deep_kernel = Deep_Kernel()
        #     with torch.no_grad():
        #         self.kernel_y = self.deep_kernel
        #         _, self.encoded_y = self.kernel_y(self.preimage_states)

        self.gram_X = None
        self.gram_Y = None
        self.EOO = None
        self.ELTO = None
        self.trans_mat0 = None
        self.inn_res = None
        self.GO = None
        self.RG = None
        self.YO = None
        self.gram_K0 = None

        # initialization
        self.Q = None
        self.R = None

        if self.noise_model == 'fixed_diag':
            self.params_q_diag = torch.empty(self.N - 1, device=self.device)
            self.params_r_diag = torch.empty(self.N, device=self.device)

        elif self.noise_model == 'fixed_bws' or self.noise_model == 'adaptive_bws' or self.noise_model == 'akf_ablation':
            # "BWS" in code == "SB" (Scalar-Block) in paper.
            # num_blocks_q / num_blocks_r: number of SB blocks k for Q and R respectively.
            # params_L: k*(k+1)/2 lower-triangular scalars for Q = L*L^T
            # params_M: k*(k+1)/2 lower-triangular scalars for R = M*M^T
            self.num_blocks_q = self.model_config.get('num_blocks_q', 2)
            self.num_blocks_r = self.model_config.get('num_blocks_r', 2)

            total_params_q = self.num_blocks_q * (self.num_blocks_q + 1) // 2
            self.params_L = torch.empty(total_params_q, device=self.device)
            total_params_r = self.num_blocks_r * (self.num_blocks_r + 1) // 2
            self.params_M = torch.empty(total_params_r, device=self.device)

        elif self.noise_model == 'ca_aekf':
            dt = 1.0  # sampling interval
            self.x_ca = torch.zeros((6, 1), device=self.device)
            self.P_ca = torch.eye(6, device=self.device)
            self.F_ca = torch.tensor([
                [1.0, 0.0, dt, 0.0, 0.5*dt**2, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.5*dt**2],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ], device=self.device)
            
            self.H_ca = torch.tensor([
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            ], device=self.device)

        elif self.noise_model == 'cv_aekf':
            dt = 1.0  
            self.x_cv = torch.zeros((4, 1), device=self.device)
            self.P_cv = torch.eye(4, device=self.device)
            
            # CV model: position + velocity only (no acceleration term)
            self.F_cv = torch.tensor([
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt ],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], device=self.device)
            
            self.H_cv = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ], device=self.device)

        self.model_learned = False

    def _construct_bws_matrix(self, factor_params_vec, total_dim, num_blocks):
        """Build an SB (Scalar-Block) noise covariance matrix A = L * L^T.

        "BWS" in code == "SB" (Scalar-Block) in the paper.
        L is a lower-triangular matrix whose (i,j)-th block is scalar_ij * I.
        The k*(k+1)/2 lower-triangular scalars come from CMA-ES (params_L_vec for Q,
        params_M_vec for R after exp-transform in the experiment script).
        """
        # Compute block dimensions; the first (total_dim % num_blocks) blocks get one
        # extra row/column to evenly distribute the remainder.
        block_dims = [total_dim // num_blocks + 1] * (total_dim % num_blocks) + \
                    [total_dim // num_blocks] * (num_blocks - (total_dim % num_blocks))

        # Build lower-triangular factor matrix L where each block is scalar * I
        factor_matrix = torch.zeros(total_dim, total_dim, device=self.device)

        param_pointer = 0
        row_start = 0
        for i in range(num_blocks):
            col_start = 0
            for j in range(i + 1):
                p_ij = factor_params_vec[param_pointer]
                param_pointer += 1

                dim_i, dim_j = block_dims[i], block_dims[j]
                scalar_block = p_ij * torch.eye(dim_i, dim_j, device=self.device)

                factor_matrix[row_start:row_start+dim_i, col_start:col_start+dim_j] = scalar_block
                col_start += dim_j
            row_start += dim_i

        # A = L * L^T  (guaranteed PSD by construction)
        bws_matrix = factor_matrix @ factor_matrix.T
        return bws_matrix

    def _decompose_bws_matrix(self, bws_matrix, num_blocks):
        """Extract the lower-triangular SB factor L from an SB (Scalar-Block) matrix.

        Used during online adaptation (noise_model='adaptive_bws') to convert an
        updated target covariance back into the k*(k+1)/2 factor scalars stored in
        self.params_L / self.params_M.  Falls back to spectral rectification
        (eigenvalue clipping to 1e-6) if the scalar coefficient matrix is not
        strictly positive definite.
        """
        total_dim = bws_matrix.shape[0]

        # Extract the k×k scalar coefficient matrix from the top-left corner of each block
        block_dims = [total_dim // num_blocks + 1] * (total_dim % num_blocks) + \
                    [total_dim // num_blocks] * (num_blocks - (total_dim % num_blocks))

        scalar_coeff_matrix = torch.zeros(num_blocks, num_blocks, device=self.device)
        end_indices = torch.cumsum(torch.tensor(block_dims), dim=0)
        start_indices = torch.cat((torch.tensor([0]), end_indices[:-1]))

        for i in range(num_blocks):
            for j in range(i + 1):
                scalar_val = bws_matrix[start_indices[i], start_indices[j]]
                scalar_coeff_matrix[i, j] = scalar_val
                if i != j:
                    scalar_coeff_matrix[j, i] = scalar_val

        # Cholesky decomposition of the k×k coefficient matrix with eigenvalue-clipping fallback
        try:
            # Optimistic path: direct Cholesky (succeeds when matrix is strictly PD)
            factor_coeff_matrix = torch.linalg.cholesky(scalar_coeff_matrix)

        except torch.linalg.LinAlgError:
            # Fallback: matrix is semi-PD or indefinite — clip negative eigenvalues
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(scalar_coeff_matrix)
                clipped_eigenvalues = torch.clamp(eigenvalues, min=1e-6)
                a_positive_definite = eigenvectors @ torch.diag(clipped_eigenvalues) @ eigenvectors.T
                factor_coeff_matrix = torch.linalg.cholesky(a_positive_definite)

            except torch.linalg.LinAlgError as e_final:
                print(f"FATAL: Cholesky failed even after eigenvalue clipping: {e_final}")
                factor_coeff_matrix = torch.eye(num_blocks, device=self.device)

        # Return the lower-triangular entries as a flat parameter vector
        indices = torch.tril_indices(row=num_blocks, col=num_blocks, offset=0)
        flat_factor_params = factor_coeff_matrix[indices[0], indices[1]]
        return flat_factor_params

        # 2. 对系数矩阵进行Cholesky分解
        # try:
        #     factor_coeff_matrix = torch.linalg.cholesky(scalar_coeff_matrix)
        # except torch.linalg.LinAlgError:
        #     print("Warning: Cholesky decomposition failed on coefficient matrix. Adding regularization.")
        #     a_reg = scalar_coeff_matrix + 1e-3 * torch.eye(num_blocks, device=self.device)
        #     factor_coeff_matrix = torch.linalg.cholesky(a_reg)

        # # 3. 用分解出的因子系数，重构完整的因子矩阵
        # reconstructed_factor = torch.zeros_like(bws_matrix)
        # row_start = 0
        # for i in range(num_blocks):
        #     col_start = 0
        #     for j in range(i + 1):
        #         p_ij = factor_coeff_matrix[i, j] # 从分解出的系数矩阵中取值
        #         dim_i, dim_j = block_dims[i], block_dims[j]
        #         scalar_block = p_ij * torch.eye(dim_i, dim_j, device=self.device)
        #         reconstructed_factor[row_start:row_start+dim_i, col_start:col_start+dim_j] = scalar_block
        #         col_start += dim_j
        #     row_start += dim_i
            
        # return reconstructed_factor

    def learn_model(self, **kwargs):
        """
        Constructs noise matrices Q and R based on the current noise_model and
        parameters from the optimizer, then learns the system operators ELTO and EOO.
        """
        # --- 1. Extract common ELTO regularization hyperparameters ---
        eps_t = kwargs.get('eps_t', np.exp(-6))
        eps_o = kwargs.get('eps_o', np.exp(-10))

        # --- 2. Build Q and R according to noise_model ---
        if self.noise_model == 'static':
            # eps_q is a single scalar noise parameter optimized by CMA-ES
            eps_q = kwargs['eps_q']
            self.Q = 1e-5 * torch.eye(self.N - 1, device=self.device)
            self.R = self.N * eps_q * torch.eye(self.N, device=self.device)

        elif self.noise_model == 'fixed_diag':
            # Decorators merge 'eq_...' args into 'eps_q_vec_k' and 'er_...' into 'eps_r_vec_k'
            params_q_k = kwargs['eps_q_vec_k']
            params_r_k = kwargs['eps_r_vec_k']

            # Build piecewise-constant diagonal covariance vectors from k block-scalar values
            N_trans = self.N - 1
            k_q = len(params_q_k)
            chunk_size_q = N_trans // k_q
            repeats_q = [chunk_size_q] * k_q
            if k_q > 0 and N_trans > 0:
                repeats_q[-1] += N_trans % k_q
            diag_vector_q = torch.from_numpy(np.repeat(params_q_k, repeats_q)).float().to(self.device)

            N_obs = self.N
            k_r = len(params_r_k)
            chunk_size_r = N_obs // k_r
            repeats_r = [chunk_size_r] * k_r
            if k_r > 0 and N_obs > 0:
                repeats_r[-1] += N_obs % k_r
            diag_vector_r = torch.from_numpy(np.repeat(params_r_k, repeats_r)).float().to(self.device)

            self.Q = torch.diag(diag_vector_q)
            self.R = torch.diag(diag_vector_r)

        elif self.noise_model == 'fixed_bws' or self.noise_model == 'adaptive_bws' or self.noise_model == 'akf_ablation':
            # "BWS" in code == "SB" (Scalar-Block) in paper.
            # Decorators merge 'param_L_...' -> 'params_L_vec'  (Cholesky factor of Q)
            #                   'param_M_...' -> 'params_M_vec'  (Cholesky factor of R)
            # For 'adaptive_bws'/'akf_ablation' these are the *initial* factors ('init_L_...' / 'init_M_...').
            params_L_vec = torch.from_numpy(kwargs['params_L_vec']).float().to(self.device)
            params_M_vec = torch.from_numpy(kwargs['params_M_vec']).float().to(self.device)

            # Build Q_0 and R_0 as SB matrices: A = L*L^T
            self.Q = self._construct_bws_matrix(params_L_vec, self.N - 1, self.num_blocks_q)
            self.R = self._construct_bws_matrix(params_M_vec, self.N, self.num_blocks_r)

        elif self.noise_model == 'ca_aekf':
            eps_q_ca = kwargs.get('eps_q_ca', 1e-5)
            eps_r_ca = kwargs.get('eps_r_ca', 1e-5)
            self.Q = eps_q_ca * torch.eye(6, device=self.device)
            self.R = eps_r_ca * torch.eye(2, device=self.device)
            
            # Initialize state
            self.x_ca = torch.zeros((6, 1), device=self.device)
            self.P_ca = torch.eye(6, device=self.device)

            self.model_learned = True
            return None, None  # no ELTO/EOO needed for physical-space models

        elif self.noise_model == 'cv_aekf':
            eps_q_cv = kwargs.get('eps_q_cv', 1e-5)
            eps_r_cv = kwargs.get('eps_r_cv', 1e-5)
            self.Q = eps_q_cv * torch.eye(4, device=self.device)
            self.R = eps_r_cv * torch.eye(2, device=self.device)
            
            self.x_cv = torch.zeros((4, 1), device=self.device)
            self.P_cv = torch.eye(4, device=self.device)
            self.model_learned = True
            return None, None

        elif self.noise_model == 'dd_aekf':
            eps_q_dd = kwargs.get('eps_q_dd', 1e-5)
            eps_r_dd = kwargs.get('eps_r_dd', 1e-5)
            
            # Infer observation dimension from the dataset
            obs_dim = self.y.shape[1]
            
            self.Q = eps_q_dd * torch.eye(obs_dim, device=self.device)
            self.R = eps_r_dd * torch.eye(obs_dim, device=self.device)
            
            # Data-Driven System Identification
            Y_past = self.y[:-1, :].T   
            Y_future = self.y[1:, :].T  
            
            self.F_dd = Y_future @ torch.linalg.pinv(Y_past)
            self.H_dd = torch.eye(obs_dim, device=self.device)
            
            self.x_dd = self.y[0:1, :].T 
            self.P_dd = torch.eye(obs_dim, device=self.device)
            
            self.model_learned = True
            return None, None

        # --- 3. Compute Gram matrices ---
        self.gram_X = self.kernel_x(self.X.T)
        gram_X2 = self.gram_X[:, 1:self.N]
        if not self.is_deep_kernel:
            self.gram_Y = self.kernel_y(self.preimage_states)
        else:
            self.gram_Y, _ = self.kernel_y(self.preimage_states)

        # --- 4. Compute ELTO (latent transition) and EOO (observation) operators ---
        self.EOO = inv(self.gram_X + self.N * eps_o * torch.eye(self.N).T.to(self.device)) @ gram_X2
        self.ELTO = inv(self.gram_X[:self.N-1,:self.N-1] + (self.N-1)
                        * eps_t * torch.eye(self.N-1).to(self.device)).T @ self.gram_X[:self.N-1,1:self.N]

        # --- 5. Auxiliary matrices for update step and pre-image ---
        self.GO = self.gram_Y @ self.EOO
        if self.is_deep_kernel:
            self.YO = self.encoded_y.T @ self.EOO
        else:
            self.YO = self.preimage_states.T @ self.EOO

        # Initial embedding for t=0 transition
        X0 = torch.randn(self.n_x0, self.d).to(self.device)
        self.gram_K0 = self.kernel_x(self.X[:,:self.N-1].T, X0)
        self.trans_mat0 = inv(self.gram_X[:self.N-1, :self.N-1] + (self.N-1)
                              * eps_t * torch.eye(self.N-1).to(self.device)).T @ self.gram_K0

        self.model_learned = True
        return self.ELTO, self.EOO

    def update_step(self, m, cov, yt):

        if self.is_deep_kernel == False:
            g_Y = self.kernel_y(self.preimage_states, yt)
        else:
            with torch.no_grad():
                _, encoded_preimage = self.kernel_y(self.preimage_states)
                _, encoded_input = self.kernel_y(yt)
                g_Y = _rbf_kernel(encoded_preimage, encoded_input)

        # kernel Kalman gain operator
        O_cov = self.EOO @ cov
        G_denominator_T = O_cov @ self.GO.T + self.R
        G = self.solve_or_pinv(G_denominator_T, O_cov).T

        # Update mean and covariance
        d = g_Y - self.GO @ m
        m_posterior = m + G @ d
        cov_posterior = cov - G @ self.GO @ cov

        # Calculate measurement residual
        v = g_Y - self.GO @ m_posterior

        return m_posterior, cov_posterior, v, d, G

    def predict_step(self, m, cov):
        # predict step, aka transition update
        m = self.ELTO @ m

        if cov is not None:
            # --- MODIFIED: The prediction logic is universal. ---
            # It always uses self.Q, which has been correctly set in learn_model.
            # No if/else branches are needed here.
            cov = self.ELTO @ cov @ self.ELTO.T + self.Q
            cov = 0.5 * (cov + cov.T)
        
        # --- NEW: Placeholder for adaptive Q update ---
        # Note: A per-step update for Q is usually done here, after prediction.
        # However, our windowed approach will handle this in the filter loop.
        # For now, we follow the structure from update_step.
        if self.noise_model == 'adaptive_bws':
            pass

        return m, cov

    def pre_image(self, m, cov):
        mu = self.YO @ m
        sig = self.YO @ cov @ self.YO.T
        return mu.T, sig

    def initial_transition(self):
        # N = self.N
        # n_x0 = self.n_x0

        # trans_mat0 = (inv(self.gram_X[:self.N-1, :self.N-1] + (self.N-1) * self.eps_t * torch.eye(self.N-1)).T @
        #               self.gram_K0)
        # self.learn_model()

        m_t0 = torch.ones((self.n_x0, 1)).to(self.device) / self.n_x0
        S_t0 = torch.eye(self.n_x0).to(self.device)
        m_t = self.trans_mat0 @ m_t0  # 1st transition
        S_t = self.trans_mat0 @ S_t0 @ self.trans_mat0.T + self.Q
        return m_t, S_t

    def solve_or_pinv(self, A, B, reg_lambda=1e-4):
        """Solve AX = B; falls back to regularized pseudoinverse if A is singular."""
        try:
            X = torch.linalg.solve(A, B)
        except torch._C._LinAlgError as e:
            if 'singular' in str(e):
                try:
                    A_reg = A + reg_lambda * torch.eye(A.shape[0], device=A.device)
                    X = torch.matmul(torch.linalg.pinv(A_reg), B)
                except torch._C._LinAlgError as pinv_e:
                    print(f"pinv also failed with error: {pinv_e}")
                    try:
                        U, S, V = torch.svd(A_reg)
                        S_inv = torch.diag(1.0 / (S + reg_lambda))
                        X = V @ S_inv @ U.T @ B
                    except torch._C._LinAlgError as svd_e:
                        print(f"SVD still failed with error: {svd_e}")
                        raise
            else:
                raise
        return X

    # def _project_to_bws(self, dense_target_matrix, num_blocks):
    #     """
    #     Projects a dense matrix onto the Block-wise Scalar (BWS) structure.

    #     This function performs the following steps:
    #     1.  Calculates the dimensions for each block automatically.
    #     2.  Iterates through each block of the dense input matrix.
    #     3.  For each block, it calculates a single scalar value that best
    #         represents the block's diagonal entries (we use the mean).
    #     4.  Constructs a new matrix where each block is a scalar matrix
    #         (scalar_value * Identity), thus conforming to the BWS structure.

    #     Args:
    #         dense_target_matrix (torch.Tensor): The dense matrix to be projected (e.g., R_target).
    #         num_blocks (int): The number of blocks (k) in the BWS structure.

    #     Returns:
    #         torch.Tensor: A new matrix that has the BWS structure.
    #     """
    #     total_dim = dense_target_matrix.shape[0]

    #     # 1. 自动计算分块维度
    #     if num_blocks > 0:
    #         block_dims = [total_dim // num_blocks + 1] * (total_dim % num_blocks) + \
    #                      [total_dim // num_blocks] * (num_blocks - (total_dim % num_blocks))
    #     else:
    #         return torch.zeros_like(dense_target_matrix) # Handle edge case

    #     # 2. 创建一个空的输出矩阵
    #     bws_matrix = torch.zeros_like(dense_target_matrix)
        
    #     # 预计算块的起始索引
    #     end_indices = torch.cumsum(torch.tensor(block_dims), dim=0)
    #     start_indices = torch.cat((torch.tensor([0]), end_indices[:-1]))

    #     # 3. 遍历所有块，提取信息并构建BWS矩阵
    #     for i in range(num_blocks):
    #         for j in range(num_blocks):
    #             # 获取当前块的行列索引
    #             row_start, row_end = start_indices[i], end_indices[i]
    #             col_start, col_end = start_indices[j], end_indices[j]
                
    #             # a. 从稠密目标矩阵中提取对应的块
    #             target_block = dense_target_matrix[row_start:row_end, col_start:col_end]
                
    #             # b. 提取该块的对角线元素
    #             diag_elements = torch.diag(target_block)
                
    #             # c. 计算一个标量来代表整个块。
    #             #    取对角线元素的平均值是一个非常稳健的选择。
    #             if len(diag_elements) > 0:
    #                 scalar_value = torch.mean(diag_elements)
    #             else:
    #                 scalar_value = 0.0 # Handle empty blocks if they occur
                
    #             # d. 创建标量矩阵块，并放入输出矩阵的正确位置
    #             #    torch.eye能正确处理非方形块
    #             dim_i, dim_j = block_dims[i], block_dims[j]
    #             scalar_block = scalar_value * torch.eye(dim_i, dim_j, device=self.device)
    #             bws_matrix[row_start:row_end, col_start:col_end] = scalar_block
        
    #     # 4. 为了严格满足对称性，取结果和其转置的平均值
    #     return 0.5 * (bws_matrix + bws_matrix.T)

    def _project_to_bws(self, dense_target_matrix, num_blocks, min_eigenvalue=1e-6):
        """
        Projects a dense matrix onto the Scalar-Block (SB) structure and strictly enforces
        Positive Definiteness via Spectral Rectification.

        This function implements the "Composite Projection Operator" described in the paper:
        1. Structural Projection: Projects dense matrix to SB structure via trace averaging.
        2. Spectral Rectification: Clamps eigenvalues to ensure SPD property.

        Args:
            dense_target_matrix (torch.Tensor): The dense matrix to be projected.
            num_blocks (int): The number of blocks (k) in the SB structure.
            min_eigenvalue (float): The threshold xi for spectral clamping (default: 1e-6).

        Returns:
            torch.Tensor: A strictly SPD matrix with SB structure.
        """
        total_dim = dense_target_matrix.shape[0]

        # 1. Compute block dimensions
        if num_blocks > 0:
            block_dims = [total_dim // num_blocks + 1] * (total_dim % num_blocks) + \
                         [total_dim // num_blocks] * (num_blocks - (total_dim % num_blocks))
        else:
            return torch.zeros_like(dense_target_matrix)

        bws_matrix = torch.zeros_like(dense_target_matrix)

        end_indices = torch.cumsum(torch.tensor(block_dims), dim=0)
        start_indices = torch.cat((torch.tensor([0]), end_indices[:-1]))

        # 2. Structural Projection: replace each block with a scalar multiple of I
        #    (trace averaging: theta_ij = tr(A_ij) / rank(A_ij), paper Eq.)
        for i in range(num_blocks):
            for j in range(num_blocks):
                row_start, row_end = start_indices[i], end_indices[i]
                col_start, col_end = start_indices[j], end_indices[j]

                target_block = dense_target_matrix[row_start:row_end, col_start:col_end]
                diag_elements = torch.diag(target_block)

                if len(diag_elements) > 0:
                    scalar_value = torch.mean(diag_elements)
                else:
                    scalar_value = 0.0

                dim_i, dim_j = block_dims[i], block_dims[j]
                scalar_block = scalar_value * torch.eye(dim_i, dim_j, device=self.device)
                bws_matrix[row_start:row_end, col_start:col_end] = scalar_block

        # 3. Symmetry Enforcement (prerequisite for eigendecomposition)
        bws_symmetric = 0.5 * (bws_matrix + bws_matrix.T)

        # 4. Spectral Rectification: A^S = V * max(Lambda, xi*I) * V^T
        #    Uses eigh (symmetric matrix) for speed and numerical stability.
        eigs, vecs = torch.linalg.eigh(bws_symmetric)
        eigs_clamped = torch.clamp(eigs, min=min_eigenvalue)
        bws_spd = vecs @ torch.diag(eigs_clamped) @ vecs.T

        return bws_spd

    def filter(self, observations):
        assert self.model_learned

        if self.noise_model == 'dd_aekf':
            T = observations.shape[0]
            obs_dim = observations.shape[1]

            y_mu_pred = torch.zeros((T, obs_dim), device=self.device)
            
            window_size = self.model_config.get('w', 20)
            alpha = self.model_config.get('alpha', 0.99)
            beta = self.model_config.get('beta', 0.99)
            
            residual_history, innovation_history, gain_history = [], [], []
            
            for t in range(T):
                y_t = observations[t:t+1, :].T
                
                x_minus = self.F_dd @ self.x_dd
                P_minus = self.F_dd @ self.P_dd @ self.F_dd.T + self.Q
                
                d_t = y_t - (self.H_dd @ x_minus)
                S = self.H_dd @ P_minus @ self.H_dd.T + self.R
                
                S_reg = S + 1e-8 * torch.eye(obs_dim, device=self.device)
                
                try:
                    inv_S = torch.linalg.inv(S_reg)
                except torch._C._LinAlgError:
                    y_mu_pred[t:, :] = 100.0  
                    break
                    
                K_t = P_minus @ self.H_dd.T @ inv_S
                
                self.x_dd = x_minus + K_t @ d_t
                
                self.P_dd = (torch.eye(obs_dim, device=self.device) - K_t @ self.H_dd) @ P_minus
                
                v_t = y_t - (self.H_dd @ self.x_dd)
                y_mu_pred[t, :] = (self.H_dd @ self.x_dd).squeeze()
                
                residual_history.append(v_t)
                innovation_history.append(d_t)
                gain_history.append(K_t)
                
                if (t + 1) % window_size == 0 and t > 0:
                    sum_vvT = torch.sum(torch.stack([v @ v.T for v in residual_history]), dim=0)
                    R_hat = (1/window_size) * sum_vvT + self.H_dd @ self.P_dd @ self.H_dd.T
                    self.R = alpha * self.R + (1 - alpha) * R_hat
                    self.R = 0.5 * (self.R + self.R.T) 
                    
                    sum_KdKd_T = torch.sum(torch.stack([(K @ d) @ (K @ d).T for K, d in zip(gain_history, innovation_history)]), dim=0)
                    Q_hat = (1/window_size) * sum_KdKd_T
                    self.Q = beta * self.Q + (1 - beta) * Q_hat
                    self.Q = 0.5 * (self.Q + self.Q.T)
                    
                    residual_history.clear()
                    innovation_history.clear()
                    gain_history.clear()

            return y_mu_pred, torch.zeros(T, obs_dim, obs_dim).to(self.device)

        # CA-AEKF: constant-acceleration Kalman filter in observation space (comparison baseline)
        if self.noise_model == 'ca_aekf':
            T = observations.shape[0]
            y_mu_pred = torch.zeros((T, 2), device=self.device)
            
            window_size = self.model_config.get('w', 20)
            alpha = self.model_config.get('alpha', 0.99)
            beta = self.model_config.get('beta', 0.99)
            
            residual_history, innovation_history, gain_history = [], [], []
            
            for t in range(T):
                y_t = observations[t:t+1, :].T  # (2, 1)
                
                # Predict
                x_minus = self.F_ca @ self.x_ca
                P_minus = self.F_ca @ self.P_ca @ self.F_ca.T + self.Q
                
                # Update
                d_t = y_t - (self.H_ca @ x_minus)
                S = self.H_ca @ P_minus @ self.H_ca.T + self.R
                S_reg = S + 1e-6 * torch.eye(2, device=self.device)
                K_t = P_minus @ self.H_ca.T @ torch.linalg.inv(S_reg)
                
                self.x_ca = x_minus + K_t @ d_t
                self.P_ca = (torch.eye(6, device=self.device) - K_t @ self.H_ca) @ P_minus
                
                v_t = y_t - (self.H_ca @ self.x_ca)
                y_mu_pred[t, :] = (self.H_ca @ self.x_ca).squeeze()
                
                # Standard AEKF update (no SB structure)
                residual_history.append(v_t)
                innovation_history.append(d_t)
                gain_history.append(K_t)
                
                if (t + 1) % window_size == 0 and t > 0:
                    sum_vvT = torch.sum(torch.stack([v @ v.T for v in residual_history]), dim=0)
                    R_hat = (1/window_size) * sum_vvT + self.H_ca @ self.P_ca @ self.H_ca.T
                    self.R = alpha * self.R + (1 - alpha) * R_hat
                    self.R = 0.5 * (self.R + self.R.T) + 1e-6 * torch.eye(2, device=self.device)
                    
                    sum_KdKd_T = torch.sum(torch.stack([(K @ d) @ (K @ d).T for K, d in zip(gain_history, innovation_history)]), dim=0)
                    Q_hat = (1/window_size) * sum_KdKd_T
                    self.Q = beta * self.Q + (1 - beta) * Q_hat
                    self.Q = 0.5 * (self.Q + self.Q.T) + 1e-6 * torch.eye(6, device=self.device)
                    
                    residual_history.clear()
                    innovation_history.clear()
                    gain_history.clear()

            return y_mu_pred, torch.zeros(T, 2, 2).to(self.device)

        # CV-AEKF: constant-velocity Kalman filter (comparison baseline)
        if self.noise_model == 'cv_aekf':
            T = observations.shape[0]
            y_mu_pred = torch.zeros((T, 2), device=self.device)
            
            window_size = self.model_config.get('w', 20)
            alpha = self.model_config.get('alpha', 0.99)
            beta = self.model_config.get('beta', 0.99)
            
            residual_history, innovation_history, gain_history = [], [], []
            
            for t in range(T):
                y_t = observations[t:t+1, :].T
                
                # Predict
                x_minus = self.F_cv @ self.x_cv
                P_minus = self.F_cv @ self.P_cv @ self.F_cv.T + self.Q
                
                # Update
                d_t = y_t - (self.H_cv @ x_minus)
                S = self.H_cv @ P_minus @ self.H_cv.T + self.R
                
                S_reg = S + 1e-8 * torch.eye(2, device=self.device)

                try:
                    inv_S = torch.linalg.inv(S_reg)
                except torch._C._LinAlgError:
                    # S lost positive-definiteness; penalize remaining timesteps
                    y_mu_pred[t:, :] = 100.0
                    break
                    
                K_t = P_minus @ self.H_cv.T @ inv_S
                
                self.x_cv = x_minus + K_t @ d_t
                self.P_cv = (torch.eye(4, device=self.device) - K_t @ self.H_cv) @ P_minus
                
                v_t = y_t - (self.H_cv @ self.x_cv)
                y_mu_pred[t, :] = (self.H_cv @ self.x_cv).squeeze()
                
                residual_history.append(v_t)
                innovation_history.append(d_t)
                gain_history.append(K_t)
                
                if (t + 1) % window_size == 0 and t > 0:
                    sum_vvT = torch.sum(torch.stack([v @ v.T for v in residual_history]), dim=0)
                    R_hat = (1/window_size) * sum_vvT + self.H_cv @ self.P_cv @ self.H_cv.T
                    self.R = alpha * self.R + (1 - alpha) * R_hat
                    # Intentionally no SPD enforcement — exposes covariance collapse under standard AEKF
                    self.R = 0.5 * (self.R + self.R.T)
                    
                    sum_KdKd_T = torch.sum(torch.stack([(K @ d) @ (K @ d).T for K, d in zip(gain_history, innovation_history)]), dim=0)
                    Q_hat = (1/window_size) * sum_KdKd_T
                    self.Q = beta * self.Q + (1 - beta) * Q_hat
                    self.Q = 0.5 * (self.Q + self.Q.T)
                    
                    residual_history.clear()
                    innovation_history.clear()
                    gain_history.clear()

            return y_mu_pred, torch.zeros(T, 2, 2).to(self.device)

        m_t, S_t = self.initial_transition()
        if self.is_deep_kernel == False:
            y_mu_pred = torch.zeros(observations.shape).to(self.device)
            y_sigma_pred = torch.zeros(observations.shape[0], observations.shape[1], observations.shape[1]).to(self.device)
        else:
            y_mu_pred = torch.zeros(observations.shape[0], 200).to(self.device)
            y_sigma_pred = torch.zeros(observations.shape[0], 200, 200).to(self.device)

        # Initialize history buffers for windowed SB covariance adaptation
        if self.noise_model == 'adaptive_bws' or self.noise_model == 'akf_ablation':
            residual_history = []
            innovation_history = []
            gain_history = []

            # Window size and exponential smoothing factors (Algorithm 1: w, α, β)
            window_size = self.model_config.get('w', 20)
            alpha = self.model_config.get('alpha', 0.99)
            beta = self.model_config.get('beta', 0.99)

            if self.noise_model == 'akf_ablation':
                # Ablation: ensure Q and R are dense tensors (no SB structure enforced)
                if hasattr(self.Q, 'is_sparse') and self.Q.is_sparse:
                    self.Q = self.Q.to_dense()
                if hasattr(self.R, 'is_sparse') and self.R.is_sparse:
                    self.R = self.R.to_dense()


        for t in range(observations.shape[0]):

            # innovation step / update step
            try:
                m_t, S_t, v_t, d_t, K_t = self.update_step(m_t, S_t, observations[t:t+1, :])
            except torch._C._LinAlgError:
                # print("yedman /n ")
                y_mu_pred = 100 * torch.ones(observations.shape).to(self.device)
                y_sigma_pred = 100 * torch.ones(observations.shape[0], observations.shape[1], observations.shape[1]).to(self.device)
                break
            # ignore use_posterior_decoding
            y_mu_pred[t, :], y_sigma_pred[t, :] = self.pre_image(m_t, S_t)

            # predict step / transition update
            m_t, S_t = self.predict_step(m_t, S_t)

            # Ablation (akf_ablation): standard AEKF update without SB projection
            if self.noise_model == 'akf_ablation':
                residual_history.append(v_t)
                innovation_history.append(d_t)
                gain_history.append(K_t)

                if (t + 1) % window_size == 0 and t > 0:
                    # --- R (measurement noise) update — direct, no SB projection ---
                    # Compute target R_k (paper Eq. 24)
                    sum_vvT = torch.sum(torch.stack([v @ v.T for v in residual_history]), dim=0)
                    R_hat = (1/window_size) * sum_vvT + self.EOO @ S_t @ self.EOO.T
                    # Exponential smooth: C_{W,t} = α*C_{W,t-1} + (1-α)*R_hat
                    self.R = alpha * self.R + (1 - alpha) * R_hat
                    # Enforce symmetry and positive-definiteness
                    self.R = 0.5 * (self.R + self.R.T)
                    self.R += 1e-6 * torch.eye(self.R.shape[0], device=self.device)

                    # --- Q (process noise) update — direct, no SB projection ---
                    # Compute target Q_k (paper Eq. 25)
                    sum_KdKd_T = torch.sum(torch.stack([(K @ d) @ (K @ d).T
                                                       for K, d in zip(gain_history, innovation_history)]), dim=0)
                    Q_hat = (1/window_size) * sum_KdKd_T
                    # Exponential smooth: C_{V,t} = β*C_{V,t-1} + (1-β)*Q_hat
                    self.Q = beta * self.Q + (1 - beta) * Q_hat
                    self.Q = 0.5 * (self.Q + self.Q.T)
                    self.Q += 1e-6 * torch.eye(self.Q.shape[0], device=self.device)

                    residual_history.clear()
                    innovation_history.clear()
                    gain_history.clear()

            # ELTO-AKF (adaptive_bws): SB-structured covariance adaptation (paper Algorithm 1)
            # "BWS" in code == "SB" (Scalar-Block) in paper.
            if self.noise_model == 'adaptive_bws':
                residual_history.append(v_t)
                innovation_history.append(d_t)
                gain_history.append(K_t)
                if (t + 1) % window_size == 0 and t > 0:
                    # --- R and M iterative update (SB structured) ---
                    # i. Compute target R_k (paper Eq. 24)
                    sum_vvT = torch.sum(torch.stack([v @ v.T for v in residual_history]), dim=0)
                    R_hat = (1/window_size) * sum_vvT + self.EOO @ S_t @ self.EOO.T

                    # ii. Project onto SB structure, then decompose to Cholesky factor
                    #     (core contribution: composite projection operator in the paper)
                    R_S_hat = self._project_to_bws(R_hat, self.num_blocks_r)
                    M_hat = self._decompose_bws_matrix(R_S_hat, self.num_blocks_r)

                    # iii. Smooth update of M factor parameters
                    self.params_M.data = alpha * self.params_M.data + (1 - alpha) * M_hat

                    # iv. Reconstruct R from updated M
                    self.R = self._construct_bws_matrix(self.params_M, self.N, self.num_blocks_r)

                    # --- Q and L iterative update (SB structured) ---
                    # i. Compute target Q_k (paper Eq. 25)
                    sum_KdKd_T = torch.sum(torch.stack([(K @ d) @ (K @ d).T
                                                       for K, d in zip(gain_history, innovation_history)]), dim=0)
                    Q_hat = (1/window_size) * sum_KdKd_T

                    # ii. Project onto SB structure, then decompose
                    Q_S_hat = self._project_to_bws(Q_hat, self.num_blocks_q)
                    L_hat = self._decompose_bws_matrix(Q_S_hat, self.num_blocks_q)

                    # iii. Smooth update of L factor parameters
                    self.params_L.data = beta * self.params_L.data + (1 - beta) * L_hat

                    # iv. Reconstruct Q from updated L
                    self.Q = self._construct_bws_matrix(self.params_L, self.N - 1, self.num_blocks_q)

                    residual_history.clear()
                    innovation_history.clear()
                    gain_history.clear()

        return y_mu_pred, y_sigma_pred

        # def filter(self, observations):
    #     assert self.model_learned
    #     m_t, S_t = self.initial_transition()
    #     if self.is_deep_kernel == False:
    #         y_mu_pred = torch.zeros(observations.shape).to(self.device)
    #         y_sigma_pred = torch.zeros(observations.shape[0], observations.shape[1], observations.shape[1]).to(self.device)
    #     else:
    #         y_mu_pred = torch.zeros(observations.shape[0], 200).to(self.device)
    #         y_sigma_pred = torch.zeros(observations.shape[0], 200, 200).to(self.device)

    #     if self.noise_model == 'adaptive_bws':
    #         # 存储窗口内的信息
    #         residual_history = []
    #         innovation_history = []
    #         gain_history = []
            
    #         # 从配置中获取窗口大小 (w, a, b)
    #         window_size = self.model_config.get('w', 20)
    #         alpha = self.model_config.get('alpha', 0.99)
    #         beta = self.model_config.get('beta', 0.99)

    #     for t in range(observations.shape[0]):

    #         # innovation step / update step
    #         try:
    #             m_t, S_t, v_t, d_t, K_t = self.update_step(m_t, S_t, observations[t:t+1, :])
    #         except torch._C._LinAlgError:
    #             # print("yedman /n ")
    #             y_mu_pred = 100 * torch.ones(observations.shape).to(self.device)
    #             y_sigma_pred = 100 * torch.ones(observations.shape[0], observations.shape[1], observations.shape[1]).to(self.device)
    #             break
    #         # ignore use_posterior_decoding
    #         y_mu_pred[t, :], y_sigma_pred[t, :] = self.pre_image(m_t, S_t)

    #         # predict step / transition update
    #         m_t, S_t = self.predict_step(m_t, S_t)

    #         # ablation
    #         # if self.noise_model == 'akf_ablation':


    #         # adaptive
    #         if self.noise_model == 'adaptive_bws':
    #             residual_history.append(v_t)
    #             innovation_history.append(d_t)
    #             gain_history.append(K_t)
    #             if (t + 1) % window_size == 0 and t > 0:
    #                 sum_vvT = torch.sum(torch.stack([v @ v.T for v in residual_history]), dim=0)
    #                 # R_hat = (1/window_size) * sum_vvT + self.EOO @ S_t @ self.GO.T # 简化的 HPH^T
    #                 R_hat = (1/window_size) * sum_vvT + self.EOO @ S_t @ self.EOO.T
                    
    #                 # ii. 投影并分解
    #                 R_S_hat = self._project_to_bws(R_hat, self.num_blocks_r)
    #                 M_hat = self._decompose_bws_matrix(R_S_hat, self.num_blocks_r)
                    
    #                 # iii. 平滑更新 M 的底层参数
    #                 self.params_M.data = alpha * self.params_M.data + (1 - alpha) * M_hat
                    
    #                 # iv. 为下一个窗口重构 R 矩阵
    #                 self.R = self._construct_bws_matrix(self.params_M, self.N, self.num_blocks_r)

    #                 # --- Q 和 L 的迭代更新 ---
    #                 # i. 计算目标 Q_k
    #                 sum_KdKd_T = torch.sum(torch.stack([(K @ d) @ (K @ d).T 
    #                                                    for K, d in zip(gain_history, innovation_history)]), dim=0)
    #                 Q_hat = (1/window_size) * sum_KdKd_T
                    
    #                 # ii. 投影并分解
    #                 Q_S_hat = self._project_to_bws(Q_hat, self.num_blocks_q)
    #                 L_hat = self._decompose_bws_matrix(Q_S_hat, self.num_blocks_q)
                    
    #                 # iii. 平滑更新 L 的底层参数
    #                 self.params_L.data = beta * self.params_L.data + (1 - beta) * L_hat
                    
    #                 # iv. 为下一个窗口重构 Q 矩阵
    #                 self.Q = self._construct_bws_matrix(self.params_L, self.N - 1, self.num_blocks_q)
                    
    #                 # c. 为下一个窗口清空历史记录
    #                 residual_history.clear()
    #                 innovation_history.clear()
    #                 gain_history.clear()

    #     return y_mu_pred, y_sigma_pred