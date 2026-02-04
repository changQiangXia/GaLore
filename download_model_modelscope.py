"""
使用 ModelScope 下载 Qwen2-0.5B-Instruct 模型（国内镜像，速度快）
"""
import os
from modelscope import snapshot_download

model_id = "qwen/Qwen2-0.5B-Instruct"
cache_dir = "./models_cache"

print(f"开始从 ModelScope 下载模型: {model_id}")
print(f"保存路径: {cache_dir}")

# 下载模型
model_dir = snapshot_download(
    model_id,
    cache_dir=cache_dir
)

print(f"\n✅ 下载完成！")
print(f"模型路径: {model_dir}")
