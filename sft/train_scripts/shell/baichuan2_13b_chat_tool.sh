 #! /bin/bash

# WORLD_SIZE=1
# MASTER_PORT=6000 #! /bin/bash

git config --global http.proxy http://10.202.1.3:18000
git config --global https.proxy http://10.202.1.3:18000
pip install rouge_chinese nltk jieba datasets fschat==0.2.23 transformers==4.31.0 deepspeed==0.10.0 accelerate==0.21.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

GPUS_PER_NODE=8
is_master=${MASTER-"0"}
if [[ $is_master -eq 1 ]];then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $HOSTNAME --master_port $MASTER_PORT"
else
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
fi

ROOT_DIR="/ArithmeticGPT/sft"
DATA_ARGS="--data_path /ArithmeticGPT/sft/train_scripts/config/OP/cal_config_v1.json"
output_dir="/ArithmeticGPT/sft/models/baichuan2_13b_chat_op_v1"
model_name_or_path="/ArithmeticGPT/sft/models/Baichuan2-13B-Chat/"


python -m torch.distributed.launch $DISTRIBUTED_ARGS \
    $ROOT_DIR/train_scripts/train_baichuan2_chat.py \
    --model_name_or_path $model_name_or_path \
    $DATA_ARGS \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 4 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --deepspeed "/ArithmeticGPT/sft/train_scripts/utils/glm_deepspeed.json" \
    --lazy_preprocess >$ROOT_DIR/logs/baichuan2_13b_chat_op_v1.log
