"""
清理训练残留缓存
"""
import os
import shutil
import torch
import gc

def cleanup():
    print("=" * 60)
    print("清理训练残留缓存")
    print("=" * 60)
    
    # 1. 清理 PyTorch CUDA 缓存
    if torch.cuda.is_available():
        print("\n[1/4] 清理 CUDA 缓存...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"  当前显存分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  当前显存预留: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # 2. Python 垃圾回收
    print("\n[2/4] Python 垃圾回收...")
    gc.collect()
    print("  完成")
    
    # 3. 清理 checkpoints（可选，保留模型但删除训练状态）
    checkpoints_dir = "./checkpoints"
    if os.path.exists(checkpoints_dir):
        print(f"\n[3/4] 发现 checkpoints 目录")
        response = input("  是否删除 checkpoints? (y/n): ").strip().lower()
        if response == 'y':
            shutil.rmtree(checkpoints_dir)
            print("  checkpoints 已删除")
        else:
            print("  保留 checkpoints")
    
    # 4. 清理 HuggingFace 缓存（可选）
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(hf_cache):
        print(f"\n[4/4] 发现 HuggingFace 缓存: {hf_cache}")
        response = input("  是否清理 HuggingFace 缓存? (y/n): ").strip().lower()
        if response == 'y':
            shutil.rmtree(hf_cache)
            print("  HuggingFace 缓存已清理")
        else:
            print("  保留 HuggingFace 缓存")
    
    print("\n" + "=" * 60)
    print("清理完成！")
    print("=" * 60)

if __name__ == "__main__":
    cleanup()
