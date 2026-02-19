from huggingface_hub import hf_hub_download, snapshot_download
import os

repo_id = "WariHima/voxcpm-1.5-resized-large"
target_dir = "./resources/checkpoints/WariHima__voxcpm-1.5-resized-large"
snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)

repo_id = "WariHima/1.5-large-ja-rev-a"
filename = ["lora_config.json", "lora_weights.safetensors", "optimizer.pth", "scheduler.pth"]

for file in filename:
  path = hf_hub_download(
      repo_id=repo_id,
      filename=file,
      local_dir="./resources/checkpoints/1.5-large-ja-rev-a",
      local_dir_use_symlinks=False
  )