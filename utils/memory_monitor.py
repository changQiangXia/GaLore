"""
GPU memory monitoring utilities for low-VRAM experiments.
"""

import json
import logging
import time
from collections import defaultdict
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Track CUDA memory usage during training."""

    def __init__(self, enable_profiling: bool = True):
        self.enable_profiling = enable_profiling
        self.reset()

    def reset(self):
        """Reset internal tracking state."""
        self.peak_allocated = 0.0
        self.peak_reserved = 0.0
        self.history: List[Dict] = []
        self.step_times: List[float] = []

    def record_step(self, step: int, loss: float, optimizer_type: str = "unknown"):
        """Record one training step memory snapshot."""
        if not self.enable_profiling or not torch.cuda.is_available():
            return None

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
        """Log a concise memory summary for one step."""
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
        """Get summary metrics for report output."""
        if not self.enable_profiling or not torch.cuda.is_available():
            return {}

        if self.history:
            return {
                "peak_allocated_gb": round(self.peak_allocated, 3),
                "peak_reserved_gb": round(self.peak_reserved, 3),
                "total_steps": len(self.history),
                "final_allocated_gb": self.history[-1]["allocated_gb"],
            }

        # Fallback for very short runs where no logging interval was hit.
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3

        return {
            "peak_allocated_gb": round(max(self.peak_allocated, max_allocated), 3),
            "peak_reserved_gb": round(max(self.peak_reserved, max_reserved), 3),
            "total_steps": 0,
            "final_allocated_gb": round(allocated, 3),
            "final_reserved_gb": round(reserved, 3),
        }

    def save_report(self, output_path: str):
        """Save summary and per-step history to JSON."""
        report = {
            "summary": self.get_summary(),
            "history": self.history,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Memory report saved to {output_path}")


def log_memory_summary(model, optimizer_type: str = "unknown"):
    """Log model parameter counts and current CUDA memory usage."""
    if not torch.cuda.is_available():
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    """Compare memory metrics between optimizers."""

    def __init__(self):
        self.results = defaultdict(dict)

    def record(self, optimizer_type: str, metric: str, value: float):
        """Record one metric for one optimizer."""
        self.results[optimizer_type][metric] = value

    def compare(self) -> Dict:
        """Return a metric-by-metric comparison report."""
        if "galore" not in self.results or "adamw" not in self.results:
            return {}

        galore = self.results["galore"]
        adamw = self.results["adamw"]

        comparison = {}
        for metric in galore.keys():
            if metric not in adamw:
                continue
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
        """Print comparison in logs."""
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
