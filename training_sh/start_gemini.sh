# #!/bin/bash

# # -----------------------------
# # 四卡 JAX 训练启动脚本
# # -----------------------------

# # 禁用 CUDA Graph（多卡稳定性关键）
# export JAX_ENABLE_CUDA_GRAPH=0

# # 不预分配显存，按需使用
# export XLA_PYTHON_CLIENT_PREALLOCATE=false

# # 控制每卡显存占用
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

# # 使用平台分配器以优化多卡训练
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# # 关闭 XLA GPU 自动调优以提高稳定性 有bug时可尝试开启
# # XLA_FLAGS="--xla_gpu_autotune_level=0" 

# # 关闭严格卷积算法选择器以避免某些卷积操作的错误
# export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
# # XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"
# # 指定使用的 GPU 卡
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# # 启动训练
# uv run scripts/train.py pi05_libero_test --exp-name=my_experiment --overwrite
export ORION_CAP_001=1   
export RION_GMEM_CONTROL=v1

XLA_FLAGS="--xla_gpu_autotune_level=0" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false" \
JAX_ENABLE_CUDA_GRAPH=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
uv run scripts/train.py pi05_libero_test --exp-name=my_experiment --overwrite
