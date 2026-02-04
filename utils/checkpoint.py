"""
Checkpoint management: 断点保存与自动恢复。
支持保存：模型权重、优化器状态、GaLore投影矩阵、Epoch/Step、随机种子。
"""
import os
import json
import torch
import logging
import glob
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    断点管理器：处理模型、优化器、GaLore状态和训练进度的保存与恢复。
    """
    def __init__(
        self,
        checkpoint_dir: str,
        save_total_limit: int = 3,
        save_optimizer: bool = True,
        save_rng_state: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit
        self.save_optimizer = save_optimizer
        self.save_rng_state = save_rng_state
        
    def save_checkpoint(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer],
        galore_state: Optional[Any],
        epoch: int,
        step: int,
        loss: float,
        config: Dict,
        is_best: bool = False
    ) -> str:
        """
        保存完整断点。
        
        Args:
            model: 模型
            optimizer: 优化器
            galore_state: GaLore状态对象
            epoch: 当前epoch
            step: 当前step
            loss: 当前loss
            config: 配置
            is_best: 是否为最佳模型
            
        Returns:
            checkpoint_path: 保存路径
        """
        checkpoint_name = f"checkpoint-epoch{epoch}-step{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # 1. 保存模型权重
        model_path = checkpoint_path / "model.safetensors"
        try:
            from safetensors.torch import save_file
            state_dict = model.state_dict()
            # 转换为可序列化的格式
            save_file(state_dict, str(model_path))
        except Exception as e:
            logger.warning(f"safetensors save failed, using torch.save: {e}")
            model_path = checkpoint_path / "pytorch_model.bin"
            torch.save(model.state_dict(), model_path)
        
        # 2. 保存优化器状态
        if self.save_optimizer and optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer.pt"
            torch.save(optimizer.state_dict(), optimizer_path)
        
        # 3. 保存GaLore状态（投影矩阵）
        if galore_state is not None:
            galore_path = checkpoint_path / "galore_state.pt"
            torch.save(galore_state.state_dict(), galore_path)
        
        # 4. 保存训练状态
        training_state = {
            "epoch": epoch,
            "step": step,
            "loss": float(loss),
            "config": config,
        }
        
        # 保存随机种子状态（单独保存为 pt 文件）
        if self.save_rng_state:
            rng_state_path = checkpoint_path / "rng_state.pt"
            rng_state = {
                "rng_state": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                rng_state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
            torch.save(rng_state, rng_state_path)
        
        state_path = checkpoint_path / "training_state.json"
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
        
        # 5. 保存当前步数标记（用于快速恢复）
        latest_path = self.checkpoint_dir / "latest"
        with open(latest_path, 'w') as f:
            f.write(checkpoint_name)
        
        # 6. 如果是最佳模型，额外保存
        if is_best:
            best_path = self.checkpoint_dir / "best"
            with open(best_path, 'w') as f:
                f.write(checkpoint_name)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # 清理旧checkpoint
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        galore_state: Optional[Any] = None,
        checkpoint_path: Optional[str] = None
    ) -> Tuple[int, int, float]:
        """
        加载断点并恢复训练状态。
        
        Args:
            model: 模型（会被原地修改）
            optimizer: 优化器（可选，会被原地修改）
            galore_state: GaLore状态（可选，会被原地修改）
            checkpoint_path: 指定路径，None则自动检测最新
            
        Returns:
            (epoch, step, loss): 恢复的训练进度
        """
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint()
            
        if checkpoint_path is None:
            logger.info("No checkpoint found, starting from scratch")
            return 0, 0, float('inf')
            
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # 1. 加载模型权重
        model_path = checkpoint_path / "model.safetensors"
        if model_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(model_path))
        else:
            model_path = checkpoint_path / "pytorch_model.bin"
            state_dict = torch.load(model_path, map_location="cpu")
        
        model.load_state_dict(state_dict)
        logger.info("Model weights loaded")
        
        # 2. 加载优化器状态
        if self.save_optimizer and optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer.pt"
            if optimizer_path.exists():
                optimizer.load_state_dict(torch.load(optimizer_path))
                logger.info("Optimizer state loaded")
        
        # 3. 加载GaLore状态
        if galore_state is not None:
            galore_path = checkpoint_path / "galore_state.pt"
            if galore_path.exists():
                galore_state.load_state_dict(torch.load(galore_path))
                logger.info("GaLore state loaded")
        
        # 4. 加载训练状态
        state_path = checkpoint_path / "training_state.json"
        with open(state_path, 'r') as f:
            training_state = json.load(f)
        
        epoch = training_state.get("epoch", 0)
        step = training_state.get("step", 0)
        loss = training_state.get("loss", float('inf'))
        
        # 恢复随机种子状态（从 pt 文件加载）
        rng_state_path = checkpoint_path / "rng_state.pt"
        if self.save_rng_state and rng_state_path.exists():
            rng_state = torch.load(rng_state_path)
            torch.set_rng_state(rng_state["rng_state"])
            if torch.cuda.is_available() and "cuda_rng_state" in rng_state:
                torch.cuda.set_rng_state_all(rng_state["cuda_rng_state"])
            logger.info("RNG state restored")
        
        logger.info(f"Resumed from epoch {epoch}, step {step}, loss {loss:.4f}")
        return epoch, step, loss
    
    def _get_latest_checkpoint(self) -> Optional[str]:
        """自动检测最新的checkpoint。"""
        latest_file = self.checkpoint_dir / "latest"
        if latest_file.exists():
            with open(latest_file, 'r') as f:
                checkpoint_name = f.read().strip()
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            if checkpoint_path.exists():
                return str(checkpoint_path)
        
        # fallback: 按修改时间排序
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        if checkpoints:
            return str(checkpoints[0])
        
        return None
    
    def _cleanup_old_checkpoints(self):
        """清理旧的checkpoint，只保留最新的N个。"""
        if self.save_total_limit <= 0:
            return
            
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for old_checkpoint in checkpoints[self.save_total_limit:]:
            logger.info(f"Removing old checkpoint: {old_checkpoint}")
            import shutil
            shutil.rmtree(old_checkpoint, ignore_errors=True)
    
    def has_checkpoint(self) -> bool:
        """检查是否存在可恢复的checkpoint。"""
        return self._get_latest_checkpoint() is not None
