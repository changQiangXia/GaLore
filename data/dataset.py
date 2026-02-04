"""
Dataset loading and processing for instruction fine-tuning.
Optimized for 4GB VRAM with max_length=512.
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    """
    Instruction-following dataset with proper formatting for Qwen2.
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        split: str = "train"
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        if split == "eval":
            # 简单的train/eval分割
            eval_size = max(1, len(self.data) // 10)
            self.data = self.data[-eval_size:]
        elif split == "train":
            eval_size = len(self.data) // 10
            self.data = self.data[:-eval_size] if eval_size > 0 else self.data
            
        logger.info(f"[{split}] Loaded {len(self.data)} samples from {data_path}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # 构建对话格式 (Qwen2 Chat Template)
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        
        # 组合完整文本
        if input_text:
            full_prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            
        full_text = full_prompt + output_text + "<|im_end|>"
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        # 构建labels (mask掉prompt部分，只计算output的loss)
        prompt_tokenized = self.tokenizer(
            full_prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = tokenized["input_ids"]
        labels = [-100] * len(prompt_tokenized["input_ids"]) + input_ids[len(prompt_tokenized["input_ids"]):]
        
        # 截断到max_length
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        
        # Padding
        pad_length = self.max_length - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            labels = labels + [-100] * pad_length
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1 if tid != self.tokenizer.pad_token_id else 0 for tid in input_ids], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def get_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    batch_size: int = 1,
    split: str = "train",
    num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader for training or evaluation.
    
    Args:
        data_path: Path to JSON data file
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (default 512 for VRAM limit)
        batch_size: Batch size (must be 1 for 4GB VRAM)
        split: "train" or "eval"
        num_workers: Number of data loading workers
    """
    dataset = InstructionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split=split
    )
    
    # 使用pin_memory加速CPU到GPU的传输
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return dataloader


def create_sample_data(output_path: str = "./data/sample_data.json", num_samples: int = 1000):
    """
    Create sample instruction-following data for testing.
    用户可以替换为真实数据集。
    """
    sample_templates = [
        {
            "instruction": "请将以下中文翻译成英文。",
            "input": "人工智能正在改变我们的生活方式。",
            "output": "Artificial intelligence is changing the way we live."
        },
        {
            "instruction": "总结以下文本的主要内容。",
            "input": "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。",
            "output": "深度学习是机器学习的一个分支，使用多层神经网络学习数据表示。"
        },
        {
            "instruction": "解释以下概念。",
            "input": "梯度下降",
            "output": "梯度下降是一种优化算法，通过沿着损失函数梯度的反方向更新参数，来最小化损失函数。"
        },
        {
            "instruction": "完成以下代码。",
            "input": "def fibonacci(n):\n    # 返回第n个斐波那契数\n    ",
            "output": "if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        },
        {
            "instruction": "回答以下问题。",
            "input": "什么是GPU显存优化？",
            "output": "GPU显存优化是指通过各种技术手段减少深度学习模型训练时的显存占用，包括梯度检查点、量化、梯度累积等方法。"
        },
    ]
    
    data = []
    for i in range(num_samples):
        template = sample_templates[i % len(sample_templates)]
        # 添加一些变化
        data.append({
            "instruction": template["instruction"],
            "input": template["input"],
            "output": template["output"]
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created {num_samples} sample data at {output_path}")


if __name__ == "__main__":
    # 测试数据加载
    create_sample_data()
