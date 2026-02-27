"""
Training script for Qwen2-0.5B with GaLore on 4GB VRAM.

Features:
- 8-bit quantization
- Layer-wise gradient projection (GaLore)
- Gradient checkpointing
- Checkpoint resume
- Memory profiling
- One-click optimizer switch (GaLore vs AdamW)
"""
import os
import sys
import yaml
import torch
import logging
import argparse
from typing import Dict, Optional
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Import custom modules
from models import load_model_and_tokenizer, attach_galore_hooks, get_galore_optimized_params
from models.galore_hook import GaLoreState
from data import get_dataloader
from data.dataset import create_sample_data
from utils import MemoryMonitor, log_memory_summary, CheckpointManager


def load_config(config_path: str = "configs/galore_config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(
    model,
    config: Dict,
    galore_state: Optional[GaLoreState] = None,
):
    """
    Create optimizer. Supports one-click switch: GaLore vs AdamW.

    Args:
        model: Model
        config: Config
        galore_state: GaLore state (required for custom GaLore fallback)

    Returns:
        optimizer, optimizer_type
    """
    optimizer_type = config["optimizer"]["type"].lower()
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    if optimizer_type == "galore":
        effective_optimizer_type = "galore"
        logger.info("=" * 60)
        logger.info("Using GaLore Optimizer (Memory Efficient)")
        logger.info("=" * 60)

        galore_cfg = config["optimizer"]["galore"]
        quant_cfg = config.get("quantization", {})
        int8_mode = quant_cfg.get("enabled", False) and quant_cfg.get("load_in_8bit", False)
        use_official_galore = not int8_mode

        if int8_mode:
            logger.warning(
                "8-bit quantization detected. Skipping galore-torch optimizer because it can "
                "break bitsandbytes Linear8bitLt weights (missing `CB`) and crash on forward. "
                "Using custom GaLore-hook fallback instead."
            )

        if use_official_galore:
            try:
                # Prefer official galore-torch when available in non-int8 mode.
                from galore_torch import GaLoreAdamW

                galore_params = []
                other_params = []

                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if param.dim() == 2 and any(t in name for t in target_modules):
                        galore_params.append(param)
                    else:
                        other_params.append(param)

                if len(galore_params) == 0:
                    raise RuntimeError(
                        "GaLore target parameter set is empty. "
                        "Set quantization.llm_int8_has_fp16_weight=true and verify trainable 2D weights."
                    )

                param_groups = [
                    {
                        "params": galore_params,
                        "rank": galore_cfg["rank"],
                        "update_proj_gap": galore_cfg["update_proj_gap"],
                        "scale": galore_cfg["scale"],
                        "proj_type": galore_cfg["proj_type"],
                    },
                    {
                        "params": other_params,
                    },
                ]

                optimizer = GaLoreAdamW(
                    param_groups,
                    lr=galore_cfg["lr"],
                    betas=tuple(galore_cfg["betas"]),
                    eps=galore_cfg["eps"],
                    weight_decay=galore_cfg["weight_decay"],
                )
                return optimizer, effective_optimizer_type

            except ImportError:
                logger.warning("galore-torch not found, using custom GaLore hook fallback")

        # Fallback: custom projected-gradient hooks + AdamW family optimizer.
        handles = attach_galore_hooks(model, config, galore_state)
        if len(handles) == 0:
            effective_optimizer_type = "adamw"
            logger.warning(
                "GaLore fallback attached 0 hooks. No projected gradients will be applied; "
                "training will continue as AdamW on current trainable params."
            )
        else:
            logger.warning(
                "Using custom GaLore fallback (%d hooks). This mode is experimental and can be slower than galore-torch.",
                len(handles),
            )

        param_groups = get_galore_optimized_params(model, galore_state, config)
        param_groups = [group for group in param_groups if len(group["params"]) > 0]
        if len(param_groups) == 0:
            raise RuntimeError("No trainable parameters found for optimizer.")

        for group in param_groups:
            group["lr"] = galore_cfg["lr"]
            group["weight_decay"] = galore_cfg["weight_decay"]

        if galore_cfg.get("eight_bit", True):
            try:
                import bitsandbytes as bnb

                optimizer = bnb.optim.AdamW8bit(
                    param_groups,
                    lr=galore_cfg["lr"],
                    betas=tuple(galore_cfg["betas"]),
                    eps=galore_cfg["eps"],
                    weight_decay=galore_cfg["weight_decay"],
                )
                logger.info("Using 8-bit AdamW with GaLore hooks")
            except ImportError:
                optimizer = torch.optim.AdamW(
                    param_groups,
                    lr=galore_cfg["lr"],
                    betas=tuple(galore_cfg["betas"]),
                    eps=galore_cfg["eps"],
                    weight_decay=galore_cfg["weight_decay"],
                )
        else:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=galore_cfg["lr"],
                betas=tuple(galore_cfg["betas"]),
                eps=galore_cfg["eps"],
                weight_decay=galore_cfg["weight_decay"],
            )

    elif optimizer_type == "adamw":
        logger.info("=" * 60)
        logger.info("Using AdamW Optimizer (Baseline)")
        logger.info("=" * 60)

        adamw_cfg = config["optimizer"]["adamw"]

        params = [p for p in model.parameters() if p.requires_grad]

        if adamw_cfg.get("eight_bit", True):
            try:
                import bitsandbytes as bnb

                optimizer = bnb.optim.AdamW8bit(
                    params,
                    lr=adamw_cfg["lr"],
                    betas=tuple(adamw_cfg["betas"]),
                    eps=adamw_cfg["eps"],
                    weight_decay=adamw_cfg["weight_decay"],
                )
                logger.info("Using 8-bit AdamW")
            except ImportError:
                logger.warning("bitsandbytes not available, using standard AdamW")
                optimizer = torch.optim.AdamW(
                    params,
                    lr=adamw_cfg["lr"],
                    betas=tuple(adamw_cfg["betas"]),
                    eps=adamw_cfg["eps"],
                    weight_decay=adamw_cfg["weight_decay"],
                )
        else:
            optimizer = torch.optim.AdamW(
                params,
                lr=adamw_cfg["lr"],
                betas=tuple(adamw_cfg["betas"]),
                eps=adamw_cfg["eps"],
                weight_decay=adamw_cfg["weight_decay"],
            )

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    if optimizer_type == "galore":
        return optimizer, effective_optimizer_type
    return optimizer, optimizer_type


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    config: Dict,
    epoch: int,
    start_step: int,
    galore_state: Optional[GaLoreState],
    memory_monitor: MemoryMonitor,
    checkpoint_manager: CheckpointManager,
    optimizer_type: str,
):
    """Train one epoch."""
    model.train()

    total_loss = 0.0
    num_micro_batches = 0
    last_step = start_step
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    logging_steps = config["training"]["logging_steps"]
    save_steps = config["training"]["save_steps"]
    empty_cache_freq = config["memory"]["empty_cache_freq"]

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = max(0, (start_step + 1) // grad_accum_steps)

    if start_step >= 0:
        logger.info(
            f"Resuming epoch {epoch + 1}: skipping first {start_step + 1} micro-batches"
        )

    for batch_idx, batch in enumerate(progress_bar):
        if batch_idx <= start_step:
            continue

        step = batch_idx
        last_step = step

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / grad_accum_steps

        loss.backward()

        total_loss += loss.item() * grad_accum_steps
        num_micro_batches += 1

        if (batch_idx + 1) % grad_accum_steps == 0:
            if optimizer_type == "galore" and galore_state is not None:
                galore_state.step += 1

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["training"]["max_grad_norm"],
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            optimizer_step += 1
            avg_loss = total_loss / max(num_micro_batches, 1)

            if logging_steps > 0 and optimizer_step % logging_steps == 0:
                memory_monitor.log_summary(optimizer_step, avg_loss, optimizer_type)
                memory_monitor.record_step(optimizer_step, avg_loss, optimizer_type)

            if save_steps > 0 and optimizer_step % save_steps == 0:
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    galore_state=galore_state,
                    epoch=epoch,
                    step=step,
                    loss=avg_loss,
                    config=config,
                )

            if (
                empty_cache_freq > 0
                and optimizer_step % empty_cache_freq == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item() * grad_accum_steps:.4f}",
                "step": step,
                "opt_step": optimizer_step,
            }
        )

    # Flush trailing gradients when final window is smaller than grad_accum_steps.
    if num_micro_batches > 0 and num_micro_batches % grad_accum_steps != 0:
        if optimizer_type == "galore" and galore_state is not None:
            galore_state.step += 1

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config["training"]["max_grad_norm"],
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1

        avg_loss = total_loss / max(num_micro_batches, 1)
        if logging_steps > 0 and optimizer_step % logging_steps == 0:
            memory_monitor.log_summary(optimizer_step, avg_loss, optimizer_type)
            memory_monitor.record_step(optimizer_step, avg_loss, optimizer_type)
        if save_steps > 0 and optimizer_step % save_steps == 0:
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                galore_state=galore_state,
                epoch=epoch,
                step=last_step,
                loss=avg_loss,
                config=config,
            )
        if (
            empty_cache_freq > 0
            and optimizer_step % empty_cache_freq == 0
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()

    avg_loss = total_loss / max(num_micro_batches, 1)
    return avg_loss


def evaluate(model, eval_loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train Qwen2-0.5B with GaLore")
    parser.add_argument(
        "--config", type=str, default="configs/galore_config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["galore", "adamw"],
        help="Override optimizer type for quick comparison",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Auto-resume from latest checkpoint"
    )
    parser.add_argument(
        "--prepare-data", action="store_true", help="Create sample data and exit"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.optimizer:
        config["optimizer"]["type"] = args.optimizer
        logger.info(f"Optimizer overridden to: {args.optimizer}")

    optimizer_type = config["optimizer"]["type"]

    if args.prepare_data:
        create_sample_data(config["data"]["train_file"], num_samples=1000)
        return

    if not os.path.exists(config["data"]["train_file"]):
        logger.info("Data file not found, creating sample data...")
        create_sample_data(config["data"]["train_file"], num_samples=1000)

    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)

    log_memory_summary(model, optimizer_type)

    logger.info("Creating data loaders...")
    eval_split_ratio = config["data"].get("eval_split_ratio", 0.1)

    train_loader = get_dataloader(
        data_path=config["data"]["train_file"],
        tokenizer=tokenizer,
        max_length=config["training"]["max_length"],
        batch_size=config["training"]["batch_size"],
        split="train",
        num_workers=config["data"]["num_workers"],
        eval_split_ratio=eval_split_ratio,
    )

    eval_loader = get_dataloader(
        data_path=config["data"]["train_file"],
        tokenizer=tokenizer,
        max_length=config["training"]["max_length"],
        batch_size=config["training"]["batch_size"],
        split="eval",
        num_workers=config["data"]["num_workers"],
        eval_split_ratio=eval_split_ratio,
    )

    galore_state = GaLoreState() if optimizer_type == "galore" else None

    optimizer, actual_optimizer_type = create_optimizer(model, config, galore_state)
    logger.info(f"Optimizer created: {actual_optimizer_type}")

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config["training"]["output_dir"],
        save_total_limit=config["checkpoint"]["save_total_limit"],
        save_optimizer=config["checkpoint"]["save_optimizer"],
        save_rng_state=config["checkpoint"]["save_rng_state"],
    )

    start_epoch = 0
    start_step = -1
    if args.resume or config["checkpoint"]["auto_resume"]:
        if checkpoint_manager.has_checkpoint():
            start_epoch, start_step, _ = checkpoint_manager.load_checkpoint(
                model=model,
                optimizer=optimizer,
                galore_state=galore_state,
            )
            if start_step >= len(train_loader) - 1:
                start_epoch += 1
                start_step = -1
            else:
                logger.info(
                    f"Resuming from epoch {start_epoch + 1}, next micro-batch index {start_step + 1}"
                )

    if start_epoch >= config["training"]["num_epochs"]:
        logger.info("Checkpoint already reached target num_epochs, nothing to train.")
        return

    memory_monitor = MemoryMonitor(
        enable_profiling=config["memory"]["enable_memory_profiling"]
    )

    logger.info("=" * 60)
    logger.info(f"Starting training with {actual_optimizer_type}")
    logger.info(f"Epochs: {start_epoch} -> {config['training']['num_epochs']}")
    logger.info(
        "Batch size: "
        f"{config['training']['batch_size']} "
        f"(effective: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']})"
    )
    logger.info(f"Max length: {config['training']['max_length']}")
    logger.info("=" * 60)

    best_eval_loss = float("inf")

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")

        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            epoch=epoch,
            start_step=start_step,
            galore_state=galore_state,
            memory_monitor=memory_monitor,
            checkpoint_manager=checkpoint_manager,
            optimizer_type=actual_optimizer_type,
        )

        logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")

        if eval_loader:
            eval_loss = evaluate(model, eval_loader, device)
            logger.info(f"Epoch {epoch + 1} - Eval Loss: {eval_loss:.4f}")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    galore_state=galore_state,
                    epoch=epoch,
                    step=len(train_loader) - 1,
                    loss=eval_loss,
                    config=config,
                    is_best=True,
                )

        # Only the first loop iteration may be resumed mid-epoch.
        start_step = -1

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)

    summary = memory_monitor.get_summary()
    logger.info(f"Peak VRAM Allocated: {summary.get('peak_allocated_gb', 'N/A')} GB")
    logger.info(f"Peak VRAM Reserved: {summary.get('peak_reserved_gb', 'N/A')} GB")

    report_path = os.path.join(
        config["training"]["output_dir"],
        f"memory_report_{actual_optimizer_type}.json",
    )
    memory_monitor.save_report(report_path)

    logger.info(f"Memory report saved to {report_path}")


if __name__ == "__main__":
    main()
