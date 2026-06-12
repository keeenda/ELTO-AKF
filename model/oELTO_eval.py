# import torch
# import numpy as np
# import os
# import math
# from matplotlib import pyplot as plt
# from torchvision import models
# from model.ELTO_Kernel import ELTO_Kernel
# # from model.ELTO_based_KF import ELTO_KF
# from model.OELTO_KF import OAKF
# # from deep_kernel_demo import Deep_Kernel
# from scipy.io import savemat
 
# def mean_squared_error(groundtruth, mu):
#     return torch.mean((groundtruth - mu) ** 2)

# # class oELTO():
# #     def __init__(self, train_input, 
# #                  validation_input, validation_groundtruth, 
# #                  test_input, test_groundtruth):
# #         super().__init__()

# #         self.device = "cuda"
# #         self._is_setup = False
# #         self.trained_x = None

# #         # self.train_model_class = ELTO_Kernel
# #         # self.eval_model_class = ELTO_KF
# #         # self.train_model = None
# #         # self.eval_model = None

# #         # 1. 直接接收并存储所有预处理好的数据集
# #         self.train_input = torch.tensor(train_input, dtype=torch.float32).to(self.device)
# #         self.validation_input = torch.tensor(validation_input, dtype=torch.float32).to(self.device)
# #         self.validation_groundtruth = torch.tensor(validation_groundtruth, dtype=torch.float32).to(self.device)
# #         self.test_input = torch.tensor(test_input, dtype=torch.float32).to(self.device)
# #         self.test_groundtruth = torch.tensor(test_groundtruth, dtype=torch.float32).to(self.device)

# #         # 2. 初始化模型和“哨兵”属性
# #         self.eval_model = None
# #         self.best_validation_loss = float('inf')
# #         self.best_params_on_validation = None
# #         self._is_operator_trained = False
# #         # print("ExperimentRunner initialized with pre-split datasets.")

# #     def train_operators(self, epochs, batch_size, window_size, d, use_okf=True):
# #         """
# #         使用 self.train_input 来训练算子。
# #         """
# #         # print("Training operators (ELTO/EOO) on the provided training set...")
# #         self.h = window_size
        
# #         train_model = ELTO_Kernel(self.train_input.cpu().numpy(), epochs, batch_size, window_size, d)
# #         trained_x = train_model.forward(is_deep_kernel=False)
        
# #         self.eval_model = ELTO_KF(trained_x, self.train_input.cpu().numpy(), window_size, d, use_okf=use_okf)
        
# #         self._is_operator_trained = True
# #         # print("Operators trained.")

# #     def validation(self, **kwargs):
# #         """
# #         这是传递给CMA的目标函数。它在内部的验证集上评估性能。
# #         """
# #         assert self._is_operator_trained, "Please run train_operators first."

# #         # 3. 组装最终参数字典
# #         final_params = kwargs
        
# #         # 4. 调用底层的评估模型
# #         #    注意：这里我们不再需要 elto_experiment.evaluate，而是 self.eval_model.filter
# #         #    并且需要一个损失函数来计算损失
# #         self.eval_model.learn_model(**final_params)
# #         mu, _ = self.eval_model.filter(self.validation_input)
# #         loss = mean_squared_error(self.validation_groundtruth, mu)
# #         current_loss = loss.item()

# #         # 5. "哨兵"逻辑
# #         if current_loss < self.best_validation_loss:
# #             self.best_validation_loss = current_loss
# #             self.best_params_on_validation = final_params # 保存物理参数
        
# #         # 6. 只返回一个标量损失值给CMA
# #         return current_loss

# #     def test_evaluation(self):
# #         """
# #         在所有优化完成后，使用找到的最佳超参数，
# #         在独立的测试集上进行一次最终的、无偏的评估。
# #         """
# #         assert self.best_params_on_validation is not None, "Optimization has not found any valid parameters yet."
# #         # print("\nPerforming final test on the provided unseen test set...")
        
# #         self.eval_model.learn_model(**self.best_params_on_validation)
# #         mu_final, _ = self.eval_model.filter(self.test_input)
# #         final_loss = mean_squared_error(self.test_groundtruth, mu_final)

# #         print(f"Final Test Loss: {final_loss.item():.8f}")
# #         print("Final test complete.")
        
# #         savemat('du_50_2.mat', {'mu': mu_final.cpu().numpy()})
# #         print("Final mu result saved")

# #         return final_loss.item(), mu_final.cpu().numpy()

# class OELTO():
#     def __init__(self, train_input, validation_input, validation_groundtruth, 
#                  test_input, test_groundtruth, noise_model, model_config):
#         super().__init__()

#         self.device = "cuda"
#         self._is_setup = False
#         self.trained_x = None
#         self.noise_model = noise_model
#         self.model_config = model_config if model_config is not None else {}

#         # 1. 直接接收并存储所有预处理好的数据集
#         self.train_input = torch.tensor(train_input, dtype=torch.float32).to(self.device)
#         self.validation_input = torch.tensor(validation_input, dtype=torch.float32).to(self.device)
#         self.validation_groundtruth = torch.tensor(validation_groundtruth, dtype=torch.float32).to(self.device)
#         self.test_input = torch.tensor(test_input, dtype=torch.float32).to(self.device)
#         self.test_groundtruth = torch.tensor(test_groundtruth, dtype=torch.float32).to(self.device)

#         # 2. 初始化模型和“哨兵”属性
#         self.eval_model = None
#         self.best_validation_loss = float('inf')
#         self.best_params_on_validation = None
#         self._is_operator_trained = False
#         # print("ExperimentRunner initialized with pre-split datasets.")

#     def train_operators(self, epochs, batch_size, window_size, d):
#         """
#         使用 self.train_input 来训练算子。
#         """
#         # print("Training operators (ELTO/EOO) on the provided training set...")
#         self.h = window_size

#         if self.noise_model in ['ca_aekf', 'cv_aekf']:
#             trained_x = None  
#             self.eval_model = OAKF(trained_x, self.train_input.cpu().numpy(), window_size, d, 
#                                    noise_model=self.noise_model, model_config=self.model_config)
#             self._is_operator_trained = True
#             return
        
#         if self.noise_model == 'dd_aekf':
#             trained_x = None  
#             self.eval_model = OAKF(trained_x, self.train_input.cpu().numpy(), window_size, d, 
#                                    noise_model=self.noise_model, model_config=self.model_config)
#             self._is_operator_trained = True
#             return
        
#         kernel_t = self.model_config.get('kernel_type', 'rbf') 
#         train_model = ELTO_Kernel(self.train_input.cpu().numpy(), epochs, batch_size, window_size, d, kernel_type=kernel_t)
#         # train_model = ELTO_Kernel(self.train_input.cpu().numpy(), epochs, batch_size, window_size, d)
#         trained_x = train_model.forward(is_deep_kernel=False)
        
#         self.eval_model = OAKF(trained_x, self.train_input.cpu().numpy(), window_size, d, 
#                                   noise_model=self.noise_model, model_config=self.model_config)
        
#         self._is_operator_trained = True
#         # print("Operators trained.")

#     def validation(self, **kwargs):
#         """
#         这是传递给CMA的目标函数。它在内部的验证集上评估性能。
#         """
#         assert self._is_operator_trained, "Please run train_operators first."

#         # 3. 组装最终参数字典
#         final_params = kwargs
        
#         # 4. 调用底层的评估模型
#         #    注意：这里我们不再需要 elto_experiment.evaluate，而是 self.eval_model.filter
#         #    并且需要一个损失函数来计算损失
#         self.eval_model.learn_model(**final_params)
#         mu, _ = self.eval_model.filter(self.validation_input)
#         loss = mean_squared_error(self.validation_groundtruth, mu)
#         current_loss = loss.item()

#         # 5. "哨兵"逻辑
#         if current_loss < self.best_validation_loss:
#             self.best_validation_loss = current_loss
#             self.best_params_on_validation = final_params # 保存物理参数
        
#         # 6. 只返回一个标量损失值给CMA
#         return current_loss

#     def test_evaluation(self):
#         """
#         在所有优化完成后，使用找到的最佳超参数，
#         在独立的测试集上进行一次最终的、无偏的评估。
#         """
#         assert self.best_params_on_validation is not None, "Optimization has not found any valid parameters yet."
#         # print("\nPerforming final test on the provided unseen test set...")
        
#         self.eval_model.learn_model(**self.best_params_on_validation)
#         mu_final, _ = self.eval_model.filter(self.test_input)
#         final_loss = mean_squared_error(self.test_groundtruth, mu_final)

#         # print(f"Final Test Loss: {final_loss.item():.8f}")
#         # print("Final test complete.")
        
#         # savemat('du_50_2.mat', {'mu': mu_final.cpu().numpy()})
#         # print("Final mu result saved")

#         return final_loss.item(), mu_final.cpu().numpy()

import torch
import numpy as np
import os
import math
from matplotlib import pyplot as plt
from torchvision import models
from model.ELTO_Kernel import ELTO_Kernel
# from model.ELTO_based_KF import ELTO_KF
from model.OELTO_KF import OAKF
# from deep_kernel_demo import Deep_Kernel
from scipy.io import savemat
 
def mean_squared_error(groundtruth, mu):
    return torch.mean((groundtruth - mu) ** 2)

# class oELTO():
#     def __init__(self, train_input, 
#                  validation_input, validation_groundtruth, 
#                  test_input, test_groundtruth):
#         super().__init__()

#         self.device = "cpu"
#         self._is_setup = False
#         self.trained_x = None

#         # self.train_model_class = ELTO_Kernel
#         # self.eval_model_class = ELTO_KF
#         # self.train_model = None
#         # self.eval_model = None

#         # 1. 直接接收并存储所有预处理好的数据集
#         self.train_input = torch.tensor(train_input, dtype=torch.float32).to(self.device)
#         self.validation_input = torch.tensor(validation_input, dtype=torch.float32).to(self.device)
#         self.validation_groundtruth = torch.tensor(validation_groundtruth, dtype=torch.float32).to(self.device)
#         self.test_input = torch.tensor(test_input, dtype=torch.float32).to(self.device)
#         self.test_groundtruth = torch.tensor(test_groundtruth, dtype=torch.float32).to(self.device)

#         # 2. 初始化模型和“哨兵”属性
#         self.eval_model = None
#         self.best_validation_loss = float('inf')
#         self.best_params_on_validation = None
#         self._is_operator_trained = False
#         # print("ExperimentRunner initialized with pre-split datasets.")

#     def train_operators(self, epochs, batch_size, window_size, d, use_okf=True):
#         """
#         使用 self.train_input 来训练算子。
#         """
#         # print("Training operators (ELTO/EOO) on the provided training set...")
#         self.h = window_size
        
#         train_model = ELTO_Kernel(self.train_input.cpu().numpy(), epochs, batch_size, window_size, d)
#         trained_x = train_model.forward(is_deep_kernel=False)
        
#         self.eval_model = ELTO_KF(trained_x, self.train_input.cpu().numpy(), window_size, d, use_okf=use_okf)
        
#         self._is_operator_trained = True
#         # print("Operators trained.")

#     def validation(self, **kwargs):
#         """
#         这是传递给CMA的目标函数。它在内部的验证集上评估性能。
#         """
#         assert self._is_operator_trained, "Please run train_operators first."

#         # 3. 组装最终参数字典
#         final_params = kwargs
        
#         # 4. 调用底层的评估模型
#         #    注意：这里我们不再需要 elto_experiment.evaluate，而是 self.eval_model.filter
#         #    并且需要一个损失函数来计算损失
#         self.eval_model.learn_model(**final_params)
#         mu, _ = self.eval_model.filter(self.validation_input)
#         loss = mean_squared_error(self.validation_groundtruth, mu)
#         current_loss = loss.item()

#         # 5. "哨兵"逻辑
#         if current_loss < self.best_validation_loss:
#             self.best_validation_loss = current_loss
#             self.best_params_on_validation = final_params # 保存物理参数
        
#         # 6. 只返回一个标量损失值给CMA
#         return current_loss

#     def test_evaluation(self):
#         """
#         在所有优化完成后，使用找到的最佳超参数，
#         在独立的测试集上进行一次最终的、无偏的评估。
#         """
#         assert self.best_params_on_validation is not None, "Optimization has not found any valid parameters yet."
#         # print("\nPerforming final test on the provided unseen test set...")
        
#         self.eval_model.learn_model(**self.best_params_on_validation)
#         mu_final, _ = self.eval_model.filter(self.test_input)
#         final_loss = mean_squared_error(self.test_groundtruth, mu_final)

#         print(f"Final Test Loss: {final_loss.item():.8f}")
#         print("Final test complete.")
        
#         savemat('du_50_2.mat', {'mu': mu_final.cpu().numpy()})
#         print("Final mu result saved")

#         return final_loss.item(), mu_final.cpu().numpy()

class OELTO():
    def __init__(self, train_input, validation_input, validation_groundtruth, 
                 test_input, test_groundtruth, noise_model, model_config):
        super().__init__()

        self.device = "cpu"
        self._is_setup = False
        self.trained_x = None
        self.noise_model = noise_model
        self.model_config = model_config if model_config is not None else {}

        # 1. 直接接收并存储所有预处理好的数据集
        self.train_input = torch.tensor(train_input, dtype=torch.float32).to(self.device)
        self.validation_input = torch.tensor(validation_input, dtype=torch.float32).to(self.device)
        self.validation_groundtruth = torch.tensor(validation_groundtruth, dtype=torch.float32).to(self.device)
        self.test_input = torch.tensor(test_input, dtype=torch.float32).to(self.device)
        self.test_groundtruth = torch.tensor(test_groundtruth, dtype=torch.float32).to(self.device)

        # 2. 初始化模型和“哨兵”属性
        self.eval_model = None
        self.best_validation_loss = float('inf')
        self.best_params_on_validation = None
        self._is_operator_trained = False
        # print("ExperimentRunner initialized with pre-split datasets.")

    def train_operators(self, epochs, batch_size, window_size, d):
        """
        使用 self.train_input 来训练算子。
        """
        # print("Training operators (ELTO/EOO) on the provided training set...")
        self.h = window_size

        if self.noise_model in ['ca_aekf', 'cv_aekf']:
            trained_x = None  
            self.eval_model = OAKF(trained_x, self.train_input.cpu().numpy(), window_size, d, 
                                   noise_model=self.noise_model, model_config=self.model_config)
            self._is_operator_trained = True
            return
        
        if self.noise_model == 'dd_aekf':
            trained_x = None  
            self.eval_model = OAKF(trained_x, self.train_input.cpu().numpy(), window_size, d, 
                                   noise_model=self.noise_model, model_config=self.model_config)
            self._is_operator_trained = True
            return
        
        kernel_t = self.model_config.get('kernel_type', 'rbf') 
        train_model = ELTO_Kernel(self.train_input.cpu().numpy(), epochs, batch_size, window_size, d, kernel_type=kernel_t)
        # train_model = ELTO_Kernel(self.train_input.cpu().numpy(), epochs, batch_size, window_size, d)
        trained_x = train_model.forward(is_deep_kernel=False)
        
        self.eval_model = OAKF(trained_x, self.train_input.cpu().numpy(), window_size, d, 
                                  noise_model=self.noise_model, model_config=self.model_config)
        
        self._is_operator_trained = True
        # print("Operators trained.")

    def validation(self, **kwargs):
        """
        这是传递给CMA的目标函数。它在内部的验证集上评估性能。
        """
        assert self._is_operator_trained, "Please run train_operators first."

        # 3. 组装最终参数字典
        final_params = kwargs
        
        # 4. 调用底层的评估模型
        #    注意：这里我们不再需要 elto_experiment.evaluate，而是 self.eval_model.filter
        #    并且需要一个损失函数来计算损失
        self.eval_model.learn_model(**final_params)
        mu, _ = self.eval_model.filter(self.validation_input)
        loss = mean_squared_error(self.validation_groundtruth, mu)
        current_loss = loss.item()

        # 5. "哨兵"逻辑
        if current_loss < self.best_validation_loss:
            self.best_validation_loss = current_loss
            self.best_params_on_validation = final_params # 保存物理参数
        
        # 6. 只返回一个标量损失值给CMA
        return current_loss

    def test_evaluation(self):
        """
        在所有优化完成后，使用找到的最佳超参数，
        在独立的测试集上进行一次最终的、无偏的评估。
        """
        assert self.best_params_on_validation is not None, "Optimization has not found any valid parameters yet."
        # print("\nPerforming final test on the provided unseen test set...")
        
        self.eval_model.learn_model(**self.best_params_on_validation)
        mu_final, _ = self.eval_model.filter(self.test_input)
        final_loss = mean_squared_error(self.test_groundtruth, mu_final)

        # print(f"Final Test Loss: {final_loss.item():.8f}")
        # print("Final test complete.")
        
        # savemat('du_50_2.mat', {'mu': mu_final.cpu().numpy()})
        # print("Final mu result saved")

        return final_loss.item(), mu_final.cpu().numpy()