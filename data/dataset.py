"""
Dataset loading and processing for instruction fine-tuning.
Optimized for 4GB VRAM with max_length=512.
"""

import json
import logging
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    """Instruction-following dataset with formatting for Qwen2."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        split: str = "train",
        eval_split_ratio: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        total = len(self.data)
        ratio = max(0.0, min(1.0, float(eval_split_ratio)))

        # Keep train non-empty whenever possible.
        if total > 1 and ratio > 0:
            eval_size = int(total * ratio)
            eval_size = min(max(eval_size, 1), total - 1)
        else:
            eval_size = 0

        if split == "eval":
            self.data = self.data[-eval_size:] if eval_size > 0 else []
        elif split == "train":
            self.data = self.data[:-eval_size] if eval_size > 0 else self.data
        else:
            raise ValueError(f"Unknown split: {split}")

        logger.info(
            f"[{split}] Loaded {len(self.data)} samples from {data_path} "
            f"(eval_split_ratio={ratio})"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        if input_text:
            full_prompt = (
                f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            full_prompt = (
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        full_text = full_prompt + output_text + "<|im_end|>"

        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        prompt_tokenized = self.tokenizer(
            full_prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        prompt_len = len(prompt_tokenized["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]

        pad_length = self.max_length - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
            labels = labels + [-100] * pad_length

        attention_mask = [1 if tid != self.tokenizer.pad_token_id else 0 for tid in input_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def get_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    batch_size: int = 1,
    split: str = "train",
    num_workers: int = 0,
    eval_split_ratio: float = 0.1,
) -> DataLoader:
    """Create DataLoader for train or eval split."""
    dataset = InstructionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split=split,
        eval_split_ratio=eval_split_ratio,
    )

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
    """Create sample instruction-following data for testing."""
    sample_templates = [
        {
            "instruction": "Translate the sentence to English.",
            "input": "人工智能正在改变我们的生活方式。",
            "output": "Artificial intelligence is changing the way we live.",
        },
        {
            "instruction": "Summarize the text in one sentence.",
            "input": "Deep learning is a branch of machine learning that uses multi-layer neural networks.",
            "output": "Deep learning is a machine learning method based on multi-layer neural networks.",
        },
        {
            "instruction": "Explain the concept.",
            "input": "Gradient descent",
            "output": "Gradient descent is an optimization method that updates parameters in the direction that reduces loss.",
        },
        {
            "instruction": "Complete the code.",
            "input": "def fibonacci(n):\n    # return nth fibonacci\n    ",
            "output": "if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        },
        {
            "instruction": "Answer the question.",
            "input": "What is GPU memory optimization?",
            "output": "GPU memory optimization reduces VRAM usage during training using techniques like checkpointing, quantization, and gradient accumulation.",
        },
    ]

    data = []
    for i in range(num_samples):
        template = sample_templates[i % len(sample_templates)]
        data.append(
            {
                "instruction": template["instruction"],
                "input": template["input"],
                "output": template["output"],
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Created {num_samples} sample data at {output_path}")


if __name__ == "__main__":
    create_sample_data()
