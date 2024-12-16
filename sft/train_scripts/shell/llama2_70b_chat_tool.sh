 #! /bin/bash

# Runs the "345M" parameter model
git config --global http.proxy http://10.202.1.3:18000
git config --global https.proxy http://10.202.1.3:18000
pip install rouge_chinese nltk jieba datasets fschat==0.2.23 transformers==4.31.0 deepspeed==0.10.0 accelerate==0.21.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

GPUS_PER_NODE=8
# WORLD_SIZE=1
# MASTER_PORT=6000
# RANK=0
# MASTER_ADDR="localhost"
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
is_master=${MASTER-"0"}
if [[ $is_master -eq 1 ]];then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $HOSTNAME --master_port $MASTER_PORT"
else
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
fi

ROOT_DIR="/ArithmeticGPT/sft"
DATA_ARGS="--data_path /ArithmeticGPT/sft/train_scripts/config/OP/cal_config_v1.json"
output_dir="/ArithmeticGPT/sft/models/llama2_70b_chat_op_v1"
model_name_or_path="/ArithmeticGPT/sft/models/Llama2-70B-Chat/"
# --per_device_train_batch_size 4 \
# --per_device_eval_batch_size 4 \
# --gradient_accumulation_steps 8 \

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
    $ROOT_DIR/train_scripts/train_llama2_chat.py \
    --model_name_or_path $model_name_or_path \
    $DATA_ARGS \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --deepspeed "/ArithmeticGPT/sft/train_scripts/utils/default_offload_opt_param.json" \
    --tf32 True \
    --lazy_preprocess >$ROOT_DIR/logs/llama_2_70b_chat_op_v1.log
