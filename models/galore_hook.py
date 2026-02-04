"""
GaLore (Gradient Low-Rank Projection) Hook Implementation.
核心实现：层级更新机制，确保梯度在反向传播时即刻投影并释放。

这是GaLore的核心创新：
- 不在优化器层面进行低秩投影
- 而是在每个层的backward hook中进行投影
- 这样梯度不会堆积，显存占用极低
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class GaLoreProjector:
    """
    GaLore投影器：管理低秩投影矩阵。
    """
    def __init__(
        self,
        rank: int,
        update_proj_gap: int = 200,
        scale: float = 0.25,
        proj_type: str = "std"
    ):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        
        self.ortho_matrix: Optional[torch.Tensor] = None
        self.ortho_matrix_t: Optional[torch.Tensor] = None  # 缓存转置
        self.step = 0
        
    def project(self, grad: torch.Tensor, step: int) -> torch.Tensor:
        """
        将梯度投影到低秩空间。
        
        Args:
            grad: 原始梯度 (out_features, in_features)
            step: 当前步数
            
        Returns:
            投影后的低秩梯度 (rank, in_features)
        """
        # 确保grad是2D的
        if grad.dim() > 2:
            grad = grad.view(-1, grad.shape[-1])
        
        # 转置使其符合GaLore论文 (d, m) -> SVD后 (d, r) @ (r, m)
        grad_t = grad.t()  # (m, d)
        
        # 每update_proj_gap步更新投影矩阵
        if step % self.update_proj_gap == 0 or self.ortho_matrix is None:
            self.ortho_matrix = self._get_orthogonal_matrix(grad_t, self.rank, self.proj_type)
            if self.ortho_matrix is not None:
                self.ortho_matrix_t = self.ortho_matrix.t()
        
        self.step = step
        
        if self.ortho_matrix is None:
            return grad
        
        # 投影到低秩空间: (r, d) @ (d, m) = (r, m) -> transpose to (m, r)
        low_rank_grad = self.ortho_matrix_t @ grad_t  # (r, m)
        return low_rank_grad.t() * self.scale  # (m, r)
    
    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        """
        将低秩梯度投影回原始空间。
        
        Args:
            low_rank_grad: 低秩梯度 (m, rank)
            
        Returns:
            恢复后的梯度 (out_features, in_features)
        """
        if self.ortho_matrix is None:
            return low_rank_grad
        
        # (d, r) @ (r, m) = (d, m) -> transpose to (m, d)
        grad_t = self.ortho_matrix @ low_rank_grad.t()  # (d, m)
        return grad_t.t() / self.scale  # (m, d)
    
    def _get_orthogonal_matrix(
        self, 
        weights: torch.Tensor, 
        rank: int, 
        proj_type: str
    ) -> Optional[torch.Tensor]:
        """
        通过SVD计算正交投影矩阵。
        """
        # 使用float32进行SVD以提高数值稳定性
        original_dtype = weights.dtype
        weights = weights.float()
        
        if proj_type == "std":
            # 标准投影：左奇异向量
            if weights.shape[0] >= weights.shape[1]:
                # (m, d) where m < d
                U, _, _ = torch.linalg.svd(weights, full_matrices=False)
                return U[:, :rank].to(original_dtype).contiguous()
            else:
                _, _, Vh = torch.linalg.svd(weights, full_matrices=False)
                return Vh[:rank, :].t().to(original_dtype).contiguous()
                
        elif proj_type == "reverse_std":
            # 反向投影
            if weights.shape[0] >= weights.shape[1]:
                _, _, Vh = torch.linalg.svd(weights, full_matrices=False)
                return Vh[:rank, :].t().to(original_dtype).contiguous()
            else:
                U, _, _ = torch.linalg.svd(weights, full_matrices=False)
                return U[:, :rank].to(original_dtype).contiguous()
                
        elif proj_type == "right":
            # 仅使用右奇异向量
            _, _, Vh = torch.linalg.svd(weights, full_matrices=False)
            return Vh[:rank, :].t().to(original_dtype).contiguous()
            
        elif proj_type == "left":
            # 仅使用左奇异向量
            U, _, _ = torch.linalg.svd(weights, full_matrices=False)
            return U[:, :rank].to(original_dtype).contiguous()
            
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")


class GaLoreState:
    """
    管理所有GaLore状态，包括投影矩阵。
    """
    def __init__(self):
        self.projectors: Dict[str, GaLoreProjector] = {}
        self.step = 0
        
    def get_projector(
        self, 
        name: str, 
        rank: int, 
        update_proj_gap: int,
        scale: float,
        proj_type: str
    ) -> GaLoreProjector:
        """获取或创建投影器。"""
        if name not in self.projectors:
            self.projectors[name] = GaLoreProjector(
                rank=rank,
                update_proj_gap=update_proj_gap,
                scale=scale,
                proj_type=proj_type
            )
        return self.projectors[name]
    
    def state_dict(self) -> Dict:
        """保存状态，包括投影矩阵和步数。"""
        state = {"step": self.step}
        for name, proj in self.projectors.items():
            state[name] = {
                "ortho_matrix": proj.ortho_matrix,
                "step": proj.step
            }
        return state
    
    def load_state_dict(self, state_dict: Dict):
        """加载状态。"""
        self.step = state_dict.get("step", 0)
        for name, proj_state in state_dict.items():
            if name == "step":
                continue
            if name in self.projectors:
                self.projectors[name].ortho_matrix = proj_state.get("ortho_matrix")
                self.projectors[name].step = proj_state.get("step", 0)


def attach_galore_hooks(
    model: nn.Module,
    config: Dict,
    galore_state: GaLoreState
) -> List:
    """
    为模型的可训练参数附加GaLore hooks。
    这是GaLore的核心：在backward时即刻完成投影并释放原始梯度。
    
    Args:
        model: 模型
        config: 配置
        galore_state: GaLore状态管理器
        
    Returns:
        handles: hook handles列表，用于后续移除
    """
    galore_cfg = config["optimizer"]["galore"]
    rank = galore_cfg["rank"]
    update_proj_gap = galore_cfg["update_proj_gap"]
    scale = galore_cfg["scale"]
    proj_type = galore_cfg["proj_type"]
    
    handles = []
    
    # 只处理2D权重矩阵 (线性层)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"]
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 只处理目标模块且维度为2的参数
        if param.dim() != 2 or not any(t in name for t in target_modules):
            continue
            
        # 获取或创建投影器
        projector = galore_state.get_projector(
            name, rank, update_proj_gap, scale, proj_type
        )
        
        # 创建backward hook
        def make_hook(param_name: str, proj: GaLoreProjector):
            def hook(grad: torch.Tensor) -> torch.Tensor:
                """
                Backward hook: 梯度即刻投影并释放。
                
                这是GaLore的关键：
                1. 收到梯度后立即进行低秩投影
                2. 释放原始梯度（PyTorch自动处理）
                3. 返回投影后的低秩梯度
                """
                if grad is None:
                    return None
                
                # 更新全局步数
                proj_step = galore_state.step
                
                # 投影到低秩空间
                low_rank_grad = proj.project(grad, proj_step)
                
                # 重要：返回低秩梯度，原始梯度会被自动释放
                return low_rank_grad
            
            return hook
        
        # 注册hook
        handle = param.register_hook(make_hook(name, projector))
        handles.append(handle)
        logger.debug(f"Attached GaLore hook to: {name}, shape: {param.shape}")
    
    logger.info(f"Attached {len(handles)} GaLore hooks")
    return handles


def get_galore_optimized_params(
    model: nn.Module,
    galore_state: GaLoreState,
    config: Dict
) -> List[Dict]:
    """
    为GaLore优化器准备参数组。
    
    GaLore参数组需要特殊处理：
    - 使用低秩投影的层：使用特殊的学习率
    - 其他参数（bias, norm, embedding）：正常优化
    
    Returns:
        List of param groups for optimizer
    """
    galore_params = []
    other_params = []
    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"]
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if param.dim() == 2 and any(t in name for t in target_modules):
            galore_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {
            "params": galore_params,
            "name": "galore",
        },
        {
            "params": other_params,
            "name": "other",
        }
    ]
    
    logger.info(f"GaLore params: {len(galore_params)}, Other params: {len(other_params)}")
    return param_groups


def project_back_gradients(
    model: nn.Module,
    galore_state: GaLoreState,
    config: Dict
):
    """
    在优化器step之前，将低秩梯度投影回原始空间。
    
    这是GaLore的两阶段流程：
    1. Backward hook: 梯度投影到低秩空间
    2. Optimizer step前: 低秩梯度投影回原始空间
    """
    galore_cfg = config["optimizer"]["galore"]
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"]
    
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
            
        if param.dim() == 2 and any(t in name for t in target_modules):
            if name in galore_state.projectors:
                projector = galore_state.projectors[name]
                # 将低秩梯度投影回原始空间
                param.grad.data = projector.project_back(param.grad.data)
