# CONFIG=$1
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1241}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
PRECISION=${PRECISION:-bf16}

# VMAE reconstruction
accelerate launch \
    --config-file configs/accelerator/8gpu.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    evaluate_tokenizer_mae.py \
    --config configs/imagenet/lightningdit_b_vmae_f8d16_cfg.yaml \
    --robust_exp True \

accelerate launch \
    --config-file configs/accelerator/8gpu.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    evaluate_tokenizer_mae.py \
    --epsilon 0.01 \
    --config configs/imagenet/lightningdit_b_vmae_f8d16_cfg.yaml \

accelerate launch \
    --config-file configs/accelerator/8gpu.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    evaluate_tokenizer_mae.py \
    --epsilon 0.05 \
    --config configs/imagenet/lightningdit_b_vmae_f8d16_cfg.yaml \

accelerate launch \
    --config-file configs/accelerator/8gpu.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    evaluate_tokenizer_mae.py \
    --epsilon 0.1 \
    --config configs/imagenet/lightningdit_b_vmae_f8d16_cfg.yaml \

accelerate launch \
    --config-file configs/accelerator/8gpu.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    evaluate_tokenizer_mae.py \
    --epsilon 0.2 \
    --config configs/imagenet/lightningdit_b_vmae_f8d16_cfg.yaml \

accelerate launch \
    --config-file configs/accelerator/8gpu.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    evaluate_tokenizer_mae.py \
    --epsilon 0.3 \
    --config configs/imagenet/lightningdit_b_vmae_f8d16_cfg.yaml \