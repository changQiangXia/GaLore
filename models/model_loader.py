"""
Model loading with 8-bit quantization and gradient checkpointing.
Optimized for 4GB VRAM.
"""
import torch
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Load Qwen2 model with 8-bit quantization for 4GB VRAM.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = config["model"]["name"]
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config["model"].get("trust_remote_code", True),
        padding_side="right"
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure 8-bit quantization
    quantization_config = None
    has_fp16_weight = False
    if config["quantization"]["enabled"] and config["quantization"]["load_in_8bit"]:
        has_fp16_weight = config["quantization"].get("llm_int8_has_fp16_weight", False)
        if has_fp16_weight:
            logger.warning(
                "llm_int8_has_fp16_weight=True is experimental in this stack. "
                "If you see a Linear8bitLt 'missing CB' crash, set it to False."
            )
        else:
            logger.warning(
                "llm_int8_has_fp16_weight is False: many 2D weights may be frozen under int8, "
                "which can disable GaLore target hooks."
            )
        logger.info("Enabling 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=config["quantization"].get("llm_int8_threshold", 6.0),
            llm_int8_has_fp16_weight=has_fp16_weight,
        )
    
    # Load model with optimizations
    model_kwargs = {
        "trust_remote_code": config["model"].get("trust_remote_code", True),
        "quantization_config": quantization_config,
        "device_map": "auto",  # 自动分配层到GPU/CPU
        "torch_dtype": getattr(torch, config["model"].get("torch_dtype", "bfloat16")),
    }
    
    # Remove None values
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
    
    logger.info("Loading model with settings: {}".format({k: str(v) for k, v in model_kwargs.items()}))
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    # Early compatibility check for the common int8 fp16-weight crash path.
    if (
        config["quantization"]["enabled"]
        and config["quantization"]["load_in_8bit"]
        and has_fp16_weight
    ):
        broken_linear_name = None
        for module_name, module in model.named_modules():
            if module.__class__.__name__ != "Linear8bitLt":
                continue
            weight = getattr(module, "weight", None)
            if weight is None or not hasattr(weight, "CB"):
                broken_linear_name = module_name
                break

        if broken_linear_name is not None:
            raise RuntimeError(
                "Detected incompatible int8 fp16-weight setup: "
                f"`{broken_linear_name}.weight` has no `CB` attribute. "
                "Set quantization.llm_int8_has_fp16_weight=false and retry."
            )
    
    # Enable gradient checkpointing to save VRAM
    if config["memory"]["gradient_checkpointing"]:
        logger.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        # 兼容旧版本transformers
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # 打印显存使用情况
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory allocated: {allocated:.2f} GB")
        logger.info(f"GPU memory reserved: {reserved:.2f} GB")
    
    return model, tokenizer


def get_model_memory_footprint(model) -> Dict[str, float]:
    """
    Get detailed memory footprint of the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算内存占用 (8-bit = 1 byte per param)
    param_memory_gb = total_params * 1 / 1024**3
    
    result = {
        "total_params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
        "estimated_memory_gb": param_memory_gb,
    }
    
    if torch.cuda.is_available():
        result["allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
        result["reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        result["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    
    return result
