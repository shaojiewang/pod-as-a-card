export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=10.82.177.159
export MASTER_PORT=123457
export KUBERNETES_CONTAINER_RESOURCE_GPU=8
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_IB_TIMEOUT=22
export NCCL_DEBUG=INFO
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

TP=$1
PP=$2
EP=$3
CP=$4

NPROC_PER_NODE=$(( ${TP} * ${PP} * ${EP} * ${CP} ))

export MEGATRON_PATH=/share/wangshaojie05/pai-megatron-patch-internal/
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/PAI-Megatron-LM-240718
torchrun --nnodes=1 --nproc-per-node=${NPROC_PER_NODE} \
    expert_parallel.py

