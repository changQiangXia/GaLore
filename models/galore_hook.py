"""
GaLore (Gradient Low-Rank Projection) hook utilities.

This module provides a shape-safe custom fallback when `galore_torch`
is not available.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GaLoreProjector:
    """Manage low-rank projection matrices for one parameter tensor."""

    def __init__(
        self,
        rank: int,
        update_proj_gap: int = 200,
        scale: float = 0.25,
        proj_type: str = "std",
    ):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type

        self.ortho_matrix: Optional[torch.Tensor] = None
        self.ortho_matrix_t: Optional[torch.Tensor] = None
        self.step = 0

    def project(self, grad: torch.Tensor, step: int) -> torch.Tensor:
        """
        Return a low-rank approximation of gradient with the same shape.

        PyTorch hooks must return gradients with identical shape, so this method
        projects to a low-rank subspace and immediately projects back.
        """
        original_shape = grad.shape
        if grad.dim() > 2:
            grad = grad.view(-1, grad.shape[-1])

        grad_t = grad.t()  # (in_features, out_features)

        if step % self.update_proj_gap == 0 or self.ortho_matrix is None:
            self.ortho_matrix = self._get_orthogonal_matrix(grad_t, self.rank, self.proj_type)
            if self.ortho_matrix is not None:
                self.ortho_matrix_t = self.ortho_matrix.t()

        self.step = step

        if self.ortho_matrix is None or self.ortho_matrix_t is None:
            return grad.view(original_shape)

        # Project to low-rank, then back to original space.
        low_rank_grad = self.ortho_matrix_t @ grad_t  # (rank, out_features)
        projected_grad_t = self.ortho_matrix @ low_rank_grad  # (in_features, out_features)
        projected_grad = projected_grad_t.t() * self.scale
        return projected_grad.view(original_shape)

    def project_back(self, projected_grad: torch.Tensor) -> torch.Tensor:
        """
        Backward-compatible no-op for legacy two-stage flow.

        In the shape-safe fallback path, projection-back is already done in
        `project`, so this function intentionally returns input unchanged.
        """
        return projected_grad

    def _get_orthogonal_matrix(
        self,
        weights: torch.Tensor,
        rank: int,
        proj_type: str,
    ) -> Optional[torch.Tensor]:
        """Compute orthogonal matrix from SVD."""
        original_dtype = weights.dtype
        weights = weights.float()

        if proj_type == "std":
            if weights.shape[0] >= weights.shape[1]:
                u, _, _ = torch.linalg.svd(weights, full_matrices=False)
                return u[:, :rank].to(original_dtype).contiguous()
            _, _, vh = torch.linalg.svd(weights, full_matrices=False)
            return vh[:rank, :].t().to(original_dtype).contiguous()

        if proj_type == "reverse_std":
            if weights.shape[0] >= weights.shape[1]:
                _, _, vh = torch.linalg.svd(weights, full_matrices=False)
                return vh[:rank, :].t().to(original_dtype).contiguous()
            u, _, _ = torch.linalg.svd(weights, full_matrices=False)
            return u[:, :rank].to(original_dtype).contiguous()

        if proj_type == "right":
            _, _, vh = torch.linalg.svd(weights, full_matrices=False)
            return vh[:rank, :].t().to(original_dtype).contiguous()

        if proj_type == "left":
            u, _, _ = torch.linalg.svd(weights, full_matrices=False)
            return u[:, :rank].to(original_dtype).contiguous()

        raise ValueError(f"Unknown proj_type: {proj_type}")


class GaLoreState:
    """Track all projector states for checkpoint save/load."""

    def __init__(self):
        self.projectors: Dict[str, GaLoreProjector] = {}
        self.step = 0

    def get_projector(
        self,
        name: str,
        rank: int,
        update_proj_gap: int,
        scale: float,
        proj_type: str,
    ) -> GaLoreProjector:
        """Get or create projector for parameter name."""
        if name not in self.projectors:
            self.projectors[name] = GaLoreProjector(
                rank=rank,
                update_proj_gap=update_proj_gap,
                scale=scale,
                proj_type=proj_type,
            )
        return self.projectors[name]

    def state_dict(self) -> Dict:
        """Serialize state, including projector matrices and step."""
        state = {"step": self.step}
        for name, proj in self.projectors.items():
            state[name] = {
                "ortho_matrix": proj.ortho_matrix,
                "step": proj.step,
            }
        return state

    def load_state_dict(self, state_dict: Dict):
        """Restore state in-place."""
        self.step = state_dict.get("step", 0)
        for name, proj_state in state_dict.items():
            if name == "step":
                continue
            if name in self.projectors:
                self.projectors[name].ortho_matrix = proj_state.get("ortho_matrix")
                self.projectors[name].step = proj_state.get("step", 0)
                if self.projectors[name].ortho_matrix is not None:
                    self.projectors[name].ortho_matrix_t = self.projectors[name].ortho_matrix.t()


def attach_galore_hooks(
    model: nn.Module,
    config: Dict,
    galore_state: GaLoreState,
) -> List:
    """Attach GaLore backward hooks to target 2D projection layers."""
    galore_cfg = config["optimizer"]["galore"]
    rank = galore_cfg["rank"]
    update_proj_gap = galore_cfg["update_proj_gap"]
    scale = galore_cfg["scale"]
    proj_type = galore_cfg["proj_type"]

    handles = []
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() != 2 or not any(t in name for t in target_modules):
            continue

        projector = galore_state.get_projector(
            name, rank, update_proj_gap, scale, proj_type
        )

        def make_hook(proj: GaLoreProjector):
            def hook(grad: torch.Tensor) -> torch.Tensor:
                if grad is None:
                    return None
                return proj.project(grad, galore_state.step)

            return hook

        handle = param.register_hook(make_hook(projector))
        handles.append(handle)
        logger.debug(f"Attached GaLore hook to: {name}, shape: {param.shape}")

    logger.info(f"Attached {len(handles)} GaLore hooks")
    return handles


def get_galore_optimized_params(
    model: nn.Module,
    galore_state: GaLoreState,
    config: Dict,
) -> List[Dict]:
    """Build optimizer param groups for fallback GaLore mode."""
    del galore_state, config  # kept for API compatibility

    galore_params = []
    other_params = []

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

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
        },
    ]

    logger.info(f"GaLore params: {len(galore_params)}, Other params: {len(other_params)}")
    return param_groups


def project_back_gradients(
    model: nn.Module,
    galore_state: GaLoreState,
    config: Dict,
):
    """No-op compatibility shim for old call sites."""
    del model, galore_state, config
    return None
