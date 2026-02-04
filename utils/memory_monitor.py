"""
GPU Memory monitoring utilities for 4GB VRAM experiments.
提供显存监控和对比分析功能。
"""
import torch
import logging
import time
from typing import Dict, Optional, List
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    显存监控器：跟踪并对比不同优化器的显存占用。
    """
    def __init__(self, enable_profiling: bool = True):
        self.enable_profiling = enable_profiling
        self.reset()
        
    def reset(self):
        """重置监控状态。"""
        self.peak_allocated = 0.0
        self.peak_reserved = 0.0
        self.history = []
        self.step_times = []
        
    def record_step(self, step: int, loss: float, optimizer_type: str = "unknown"):
        """
        记录当前步的显存使用情况。
        
        Args:
            step: 当前步数
            loss: 当前loss
            optimizer_type: "galore" | "adamw"
        """
        if not self.enable_profiling or not torch.cuda.is_available():
            return
            
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)
        
        record = {
            "step": step,
            "loss": loss,
            "optimizer": optimizer_type,
            "allocated_gb": round(allocated, 3),
            "reserved_gb": round(reserved, 3),
            "max_allocated_gb": round(max_allocated, 3),
            "timestamp": time.time(),
        }
        self.history.append(record)
        
        return record
    
    def log_summary(self, step: int, loss: float, optimizer_type: str = "unknown"):
        """
        输出当前显存状态摘要。
        """
        if not torch.cuda.is_available():
            logger.info(f"Step {step} | Loss: {loss:.4f}")
            return
            
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        logger.info(
            f"[{optimizer_type.upper()}] Step {step} | "
            f"Loss: {loss:.4f} | "
            f"VRAM: {allocated:.2f}GB / {reserved:.2f}GB | "
            f"Peak: {max_allocated:.2f}GB"
        )
        
    def get_summary(self) -> Dict:
        """获取监控摘要。"""
        if not self.history:
            return {}
            
        return {
            "peak_allocated_gb": round(self.peak_allocated, 3),
            "peak_reserved_gb": round(self.peak_reserved, 3),
            "total_steps": len(self.history),
            "final_allocated_gb": self.history[-1]["allocated_gb"] if self.history else 0,
        }
    
    def save_report(self, output_path: str):
        """保存完整监控报告。"""
        report = {
            "summary": self.get_summary(),
            "history": self.history,
        }
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Memory report saved to {output_path}")


def log_memory_summary(model, optimizer_type: str = "unknown"):
    """
    打印模型和显存摘要。
    
    Args:
        model: PyTorch模型
        optimizer_type: 优化器类型标识
    """
    if not torch.cuda.is_available():
        return
        
    # 模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 显存
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    logger.info("=" * 60)
    logger.info(f"[{optimizer_type.upper()}] Memory Summary")
    logger.info("=" * 60)
    logger.info(f"Total params: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable params: {trainable_params / 1e6:.2f}M")
    logger.info(f"VRAM Allocated: {allocated:.2f} GB")
    logger.info(f"VRAM Reserved: {reserved:.2f} GB")
    logger.info(f"VRAM Peak: {max_allocated:.2f} GB")
    logger.info("=" * 60)


class OptimizerComparison:
    """
    优化器对比分析器：对比GaLore和AdamW的显存占用。
    """
    def __init__(self):
        self.results = defaultdict(dict)
        
    def record(self, optimizer_type: str, metric: str, value: float):
        """记录指标。"""
        self.results[optimizer_type][metric] = value
        
    def compare(self) -> Dict:
        """对比两种优化器。"""
        if "galore" not in self.results or "adamw" not in self.results:
            return {}
            
        galore = self.results["galore"]
        adamw = self.results["adamw"]
        
        comparison = {}
        for metric in galore.keys():
            if metric in adamw:
                galore_val = galore[metric]
                adamw_val = adamw[metric]
                saved = adamw_val - galore_val
                saved_pct = (saved / adamw_val * 100) if adamw_val > 0 else 0
                comparison[metric] = {
                    "galore": round(galore_val, 3),
                    "adamw": round(adamw_val, 3),
                    "saved_gb": round(saved, 3),
                    "saved_pct": round(saved_pct, 1),
                }
        return comparison
    
    def print_comparison(self):
        """打印对比结果。"""
        comparison = self.compare()
        if not comparison:
            logger.info("No comparison data available")
            return
            
        logger.info("=" * 60)
        logger.info("Optimizer Comparison: GaLore vs AdamW")
        logger.info("=" * 60)
        for metric, values in comparison.items():
            logger.info(f"{metric}:")
            logger.info(f"  GaLore: {values['galore']:.3f} GB")
            logger.info(f"  AdamW:  {values['adamw']:.3f} GB")
            logger.info(f"  Saved:  {values['saved_gb']:.3f} GB ({values['saved_pct']}%)")
        logger.info("=" * 60)
