"""
下载 Qwen2-0.5B-Instruct 模型到本地
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-0.5B-Instruct"
local_dir = "./models_cache/Qwen2-0.5B-Instruct"

print(f"开始下载模型: {model_name}")
print(f"保存路径: {local_dir}")

# 创建目录
os.makedirs(local_dir, exist_ok=True)

# 下载 tokenizer
print("\n[1/2] 下载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
tokenizer.save_pretrained(local_dir)
print(f"Tokenizer 已保存到: {local_dir}")

# 下载模型
print("\n[2/2] 下载模型权重...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="cpu"  # 先下载到CPU，避免显存不足
)
model.save_pretrained(local_dir)
print(f"模型已保存到: {local_dir}")

print("\n✅ 下载完成！")
print(f"本地模型路径: {local_dir}")
