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
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Import custom modules
from models import load_model_and_tokenizer, attach_galore_hooks, get_galore_optimized_params
from models.galore_hook import GaLoreState, project_back_gradients
from data import get_dataloader
from data.dataset import create_sample_data
from utils import MemoryMonitor, log_memory_summary, CheckpointManager
from utils.memory_monitor import OptimizerComparison


def load_config(config_path: str = "configs/galore_config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(
    model,
    config: Dict,
    galore_state: Optional[GaLoreState] = None
):
    """
    创建优化器，支持一键切换：GaLore vs AdamW
    
    Args:
        model: 模型
        config: 配置
        galore_state: GaLore状态（仅GaLore优化器需要）
        
    Returns:
        optimizer, optimizer_type
    """
    optimizer_type = config["optimizer"]["type"].lower()
    
    if optimizer_type == "galore":
        logger.info("=" * 60)
        logger.info("Using GaLore Optimizer (Memory Efficient)")
        logger.info("=" * 60)
        
        galore_cfg = config["optimizer"]["galore"]
        
        try:
            # 尝试使用galore-torch库
            from galore_torch import GaLoreAdamW
            
            # GaLore参数组 - 使用galore-torch的特殊参数格式
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
            
            # galore-torch的API: 传入param_groups，GaLore参数需要特殊标记
            param_groups = [
                {
                    'params': galore_params,
                    'rank': galore_cfg['rank'],
                    'update_proj_gap': galore_cfg['update_proj_gap'],
                    'scale': galore_cfg['scale'],
                    'proj_type': galore_cfg['proj_type'],
                },
                {
                    'params': other_params,
                }
            ]
            
            optimizer = GaLoreAdamW(
                param_groups,
                lr=galore_cfg['lr'],
                betas=tuple(galore_cfg['betas']),
                eps=galore_cfg['eps'],
                weight_decay=galore_cfg['weight_decay'],
            )
            
        except ImportError:
            # 回退到自定义实现
            logger.warning("galore-torch not found, using custom implementation")
            
            # 先附加hooks进行梯度投影
            attach_galore_hooks(model, config, galore_state)
            
            # 使用普通AdamW，但参数需要特殊处理
            param_groups = get_galore_optimized_params(model, galore_state, config)
            for group in param_groups:
                group['lr'] = galore_cfg['lr']
                group['weight_decay'] = galore_cfg['weight_decay']
            
            if galore_cfg.get('eight_bit', True):
                try:
                    import bitsandbytes as bnb
                    optimizer = bnb.optim.AdamW8bit(
                        param_groups,
                        lr=galore_cfg['lr'],
                        betas=tuple(galore_cfg['betas']),
                        eps=galore_cfg['eps'],
                        weight_decay=galore_cfg['weight_decay'],
                    )
                    logger.info("Using 8-bit AdamW with GaLore hooks")
                except ImportError:
                    optimizer = torch.optim.AdamW(
                        param_groups,
                        lr=galore_cfg['lr'],
                        betas=tuple(galore_cfg['betas']),
                        eps=galore_cfg['eps'],
                        weight_decay=galore_cfg['weight_decay'],
                    )
            else:
                optimizer = torch.optim.AdamW(
                    param_groups,
                    lr=galore_cfg['lr'],
                    betas=tuple(galore_cfg['betas']),
                    eps=galore_cfg['eps'],
                    weight_decay=galore_cfg['weight_decay'],
                )
    
    elif optimizer_type == "adamw":
        logger.info("=" * 60)
        logger.info("Using AdamW Optimizer (Baseline)")
        logger.info("=" * 60)
        
        adamw_cfg = config["optimizer"]["adamw"]
        
        # 收集所有可训练参数
        params = [p for p in model.parameters() if p.requires_grad]
        
        if adamw_cfg.get('eight_bit', True):
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    params,
                    lr=adamw_cfg['lr'],
                    betas=tuple(adamw_cfg['betas']),
                    eps=adamw_cfg['eps'],
                    weight_decay=adamw_cfg['weight_decay'],
                )
                logger.info("Using 8-bit AdamW")
            except ImportError:
                logger.warning("bitsandbytes not available, using standard AdamW")
                optimizer = torch.optim.AdamW(
                    params,
                    lr=adamw_cfg['lr'],
                    betas=tuple(adamw_cfg['betas']),
                    eps=adamw_cfg['eps'],
                    weight_decay=adamw_cfg['weight_decay'],
                )
        else:
            optimizer = torch.optim.AdamW(
                params,
                lr=adamw_cfg['lr'],
                betas=tuple(adamw_cfg['betas']),
                eps=adamw_cfg['eps'],
                weight_decay=adamw_cfg['weight_decay'],
            )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
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
    optimizer_type: str
):
    """
    训练一个epoch。
    """
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    logging_steps = config['training']['logging_steps']
    save_steps = config['training']['save_steps']
    empty_cache_freq = config['memory']['empty_cache_freq']
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        step = start_step + batch_idx
        
        # 移动数据到GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss / grad_accum_steps
        
        # 反向传播
        loss.backward()
        
        total_loss += loss.item() * grad_accum_steps
        
        # 梯度累积
        if (batch_idx + 1) % grad_accum_steps == 0:
            # GaLore特殊处理：在step前将低秩梯度投影回原始空间
            if optimizer_type == "galore" and galore_state is not None:
                project_back_gradients(model, galore_state, config)
                galore_state.step += 1
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['max_grad_norm']
            )
            
            # 优化器step
            optimizer.step()
            optimizer.zero_grad()
            
            num_batches += 1
            
            # 显存监控和日志
            if step % logging_steps == 0:
                avg_loss = total_loss / num_batches if num_batches > 0 else loss.item()
                memory_monitor.log_summary(step, avg_loss, optimizer_type)
                memory_monitor.record_step(step, avg_loss, optimizer_type)
            
            # 保存checkpoint
            if step % save_steps == 0:
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    galore_state=galore_state,
                    epoch=epoch,
                    step=step,
                    loss=total_loss / num_batches if num_batches > 0 else loss.item(),
                    config=config
                )
            
            # 清空缓存
            if step % empty_cache_freq == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item() * grad_accum_steps:.4f}',
            'step': step
        })
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def evaluate(model, eval_loader, device):
    """评估模型。"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train Qwen2-0.5B with GaLore")
    parser.add_argument("--config", type=str, default="configs/galore_config.yaml",
                        help="Path to config file")
    parser.add_argument("--optimizer", type=str, choices=["galore", "adamw"],
                        help="Override optimizer type for quick comparison")
    parser.add_argument("--resume", action="store_true",
                        help="Auto-resume from latest checkpoint")
    parser.add_argument("--prepare-data", action="store_true",
                        help="Create sample data and exit")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override optimizer if specified
    if args.optimizer:
        config["optimizer"]["type"] = args.optimizer
        logger.info(f"Optimizer overridden to: {args.optimizer}")
    
    optimizer_type = config["optimizer"]["type"]
    
    # Prepare data
    if args.prepare_data:
        create_sample_data(config['data']['train_file'], num_samples=1000)
        return
    
    # Check if data exists
    if not os.path.exists(config['data']['train_file']):
        logger.info("Data file not found, creating sample data...")
        create_sample_data(config['data']['train_file'], num_samples=1000)
    
    # Set seed
    seed = config['training']['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # 打印初始显存
    log_memory_summary(model, optimizer_type)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = get_dataloader(
        data_path=config['data']['train_file'],
        tokenizer=tokenizer,
        max_length=config['training']['max_length'],
        batch_size=config['training']['batch_size'],
        split="train",
        num_workers=config['data']['num_workers']
    )
    
    eval_loader = get_dataloader(
        data_path=config['data']['train_file'],
        tokenizer=tokenizer,
        max_length=config['training']['max_length'],
        batch_size=config['training']['batch_size'],
        split="eval",
        num_workers=config['data']['num_workers']
    )
    
    # Initialize GaLore state (for GaLore optimizer)
    galore_state = GaLoreState() if optimizer_type == "galore" else None
    
    # Create optimizer
    optimizer, actual_optimizer_type = create_optimizer(model, config, galore_state)
    logger.info(f"Optimizer created: {actual_optimizer_type}")
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config['training']['output_dir'],
        save_total_limit=config['checkpoint']['save_total_limit'],
        save_optimizer=config['checkpoint']['save_optimizer'],
        save_rng_state=config['checkpoint']['save_rng_state']
    )
    
    # Auto-resume
    start_epoch = 0
    start_step = 0
    if args.resume or config['checkpoint']['auto_resume']:
        if checkpoint_manager.has_checkpoint():
            start_epoch, start_step, _ = checkpoint_manager.load_checkpoint(
                model=model,
                optimizer=optimizer,
                galore_state=galore_state
            )
            start_epoch += 1  # 从下一个epoch开始
    
    # Memory monitor
    memory_monitor = MemoryMonitor(enable_profiling=config['memory']['enable_memory_profiling'])
    
    # Training loop
    logger.info("=" * 60)
    logger.info(f"Starting training with {actual_optimizer_type}")
    logger.info(f"Epochs: {start_epoch} -> {config['training']['num_epochs']}")
    logger.info(f"Batch size: {config['training']['batch_size']} (effective: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']})")
    logger.info(f"Max length: {config['training']['max_length']}")
    logger.info("=" * 60)
    
    best_eval_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
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
            optimizer_type=actual_optimizer_type
        )
        
        logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
        
        # Evaluate
        if eval_loader:
            eval_loss = evaluate(model, eval_loader, device)
            logger.info(f"Epoch {epoch + 1} - Eval Loss: {eval_loss:.4f}")
            
            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    galore_state=galore_state,
                    epoch=epoch,
                    step=start_step + len(train_loader),
                    loss=eval_loss,
                    config=config,
                    is_best=True
                )
        
        # Reset step counter for next epoch
        start_step = 0
    
    # Final summary
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)
    
    summary = memory_monitor.get_summary()
    logger.info(f"Peak VRAM Allocated: {summary.get('peak_allocated_gb', 'N/A')} GB")
    logger.info(f"Peak VRAM Reserved: {summary.get('peak_reserved_gb', 'N/A')} GB")
    
    # Save memory report
    report_path = os.path.join(config['training']['output_dir'], f"memory_report_{actual_optimizer_type}.json")
    memory_monitor.save_report(report_path)
    
    logger.info(f"Memory report saved to {report_path}")


if __name__ == "__main__":
    main()
