from opencompass.models import HuggingFaceCausalLM
from opencompass.models import HuggingFace


# llama2-13b-chat_{A}
llama2_13b_chat_op_path = [104,512,2042,0,3064,0,12,52]   
llama2_13b_chat_op_models = []    
for i in range(1, len(llama2_13b_chat_op_path)+1):  
    if  llama2_13b_chat_op_path[i-1] != 0:
        llama_model_13b = dict(
            type=HuggingFaceCausalLM,
            abbr=f'llama2-13b-chat-op-v{i}',
            path=f"/ArithmeticGPT/sft/models/llama_2_13b_chat_op_cal_config_v{i}/checkpoint-{llama2_13b_chat_op_path[i-1]}",
            tokenizer_path=f"/ArithmeticGPT/sft/models/llama_2_13b_chat_op_cal_config_v{i}/checkpoint-{llama2_13b_chat_op_path[i-1]}",
            tokenizer_kwargs=dict(padding_side='left',
                                truncation_side='left',
                                use_fast=False,
                                ),
            max_out_len=2048,
            max_seq_len=2048,
            batch_size=16,
            model_kwargs=dict(device_map='auto'),
            batch_padding=False, # if false, inference with for-loop without batch padding
            run_cfg=dict(num_gpus=1, num_procs=1),
        )
        llama2_13b_chat_op_models.append(llama_model_13b)
        if i == 1:
            llama2_13b_chat_op_model_test = [llama_model_13b]
        
# llama2-13b-chat_{A/L}
llama2_13b_chat_op_general_path = [1032,1072,1124,1226,1328,1430,1532]    
llama2_13b_chat_op_general_models = []    
for i in range(1, len(llama2_13b_chat_op_general_path)+1):    
    llama_model_13b = dict(
        type=HuggingFaceCausalLM,
        abbr=f'llama2-13b-chat-op-general-v{i}',
        path=f"/ArithmeticGPT/sft/models/llama_2_13b_chat_op_general_cal_config_v{i}/checkpoint-{llama2_13b_chat_op_general_path[i-1]}",
        tokenizer_path=f"/ArithmeticGPT/sft/models/llama_2_13b_chat_op_general_cal_config_v{i}/checkpoint-{llama2_13b_chat_op_general_path[i-1]}",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
    llama2_13b_chat_op_general_models.append(llama_model_13b)
    
# llama2-13b-chat_{A/L/M}
llama2_13b_chat_op_general_mwp_path = [1134, 1174, 1226, 1634, 2144] 
llama2_13b_chat_op_general_mwp_models = []    
for i in range(1, len(llama2_13b_chat_op_general_mwp_path)+1):    
    llama_model_13b = dict(
        type=HuggingFaceCausalLM,
        abbr=f'llama2-13b-chat-op-general-wp-v{i}',
        path=f"/ArithmeticGPT/sft/models/llama_2_13b_chat_op_general_wp_cal_config_v{i}/checkpoint-{llama2_13b_chat_op_general_mwp_path[i-1]}",
        tokenizer_path=f"/ArithmeticGPT/sft/models/llama_2_13b_chat_op_general_wp_cal_config_v{i}/checkpoint-{llama2_13b_chat_op_general_mwp_path[i-1]}",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
    llama2_13b_chat_op_general_mwp_models.append(llama_model_13b)

llama2_13b_chat_tool = llama2_13b_chat_op_models + llama2_13b_chat_op_general_models + llama2_13b_chat_op_general_mwp_models
# llama2-70b-chat_{A}
llama2_70b_chat_op_path = [78,384,766,0,1725,0,8,40]    
llama2_70b_chat_op_models = []    
for i in range(1, len(llama2_70b_chat_op_path)+1):    
    if llama2_70b_chat_op_path[i-1] != 0:
        llama_model_70b = dict(
            type=HuggingFaceCausalLM,
            abbr=f'llama2-70b-chat-op-v{i}',
            path=f"/ArithmeticGPT/sft/models/llama_2_70b_base_op_cal_config_v{i}/checkpoint-{llama2_70b_chat_op_path[i-1]}",
            tokenizer_path=f"/ArithmeticGPT/sft/models/llama_2_70b_base_op_cal_config_v{i}/checkpoint-{llama2_70b_chat_op_path[i-1]}",
            tokenizer_kwargs=dict(padding_side='left',
                                truncation_side='left',
                                use_fast=False,
                                ),
            max_out_len=2048,
            max_seq_len=2048,
            batch_size=16,
            model_kwargs=dict(device_map='auto'),
            batch_padding=False, # if false, inference with for-loop without batch padding
            run_cfg=dict(num_gpus=2, num_procs=1),
        )
        llama2_70b_chat_op_models.append(llama_model_70b)
        
# llama2-70b-chat_{A/L}
llama2_70b_chat_op_general_path = [774,804,844,920, 996, 1072, 1150] 
llama2_70b_chat_op_general_models = []    
for i in range(1, len(llama2_70b_chat_op_general_path)+1):    
    llama_model_70b = dict(
        type=HuggingFaceCausalLM,
        abbr=f'llama2-70b-chat-op-general-v{i}',
        path=f"/ArithmeticGPT/sft/models/llama_2_70b_chat_op_general_cal_config_v{i}/checkpoint-{llama2_70b_chat_op_general_path[i-1]}",
        tokenizer_path=f"/ArithmeticGPT/sft/models/llama_2_70b_chat_op_general_cal_config_v{i}/checkpoint-{llama2_70b_chat_op_general_path[i-1]}",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
    llama2_70b_chat_op_general_models.append(llama_model_70b)
    
# llama2-70b-chat_{A/L/M}
llama2_70b_chat_op_general_mwp_path = [812, 844, 882, 1188, 1570] 
llama2_70b_chat_op_general_mwp_models = []    
for i in range(1, len(llama2_70b_chat_op_general_mwp_path)+1):    
    llama_model_70b = dict(
        type=HuggingFaceCausalLM,
        abbr=f'llama2-70b-chat-op-general-wp-v{i}',
        path=f"/ArithmeticGPT/sft/models/llama_2_70b_chat_op_general_wp_cal_config_v{i}/checkpoint-{llama2_70b_chat_op_general_mwp_path[i-1]}",
        tokenizer_path=f"/ArithmeticGPT/sft/models/llama_2_70b_chat_op_general_wp_cal_config_v{i}/checkpoint-{llama2_70b_chat_op_general_mwp_path[i-1]}",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=16,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=2, num_procs=1),
    )
    llama2_70b_chat_op_general_mwp_models.append(llama_model_70b)
    
llama2_70b_chat_tool = llama2_70b_chat_op_models + llama2_70b_chat_op_general_models + llama2_70b_chat_op_general_mwp_models

# baichuan2-13b-chat_{A}
baichuan2_13b_chat_op_path = [104,512,1022,0,1532,0,12,52]    
baichuan2_13b_chat_op_models = []    
for i in range(1, len(baichuan2_13b_chat_op_path)+1):    
    if baichuan2_13b_chat_op_path[i-1] != 0:
        baichuan2_model_13b = dict(
            type=HuggingFaceCausalLM,
            abbr=f'baichuan2-13b-chat-op-v{i}',
            path=f"/ArithmeticGPT/sft/models/baichuan2_13b_base_op_cal_config_v{i}/checkpoint-{baichuan2_13b_chat_op_path[i-1]}",
            tokenizer_path=f"/ArithmeticGPT/sft/models/baichuan2_13b_base_op_cal_config_v{i}/checkpoint-{baichuan2_13b_chat_op_path[i-1]}",
            tokenizer_kwargs=dict(padding_side='left',
                                truncation_side='left',
                                use_fast=False,
                                trust_remote_code=True,
                                ),
            max_out_len=2048,
            max_seq_len=2048,
            batch_size=512,
            model_kwargs=dict(trust_remote_code=True, device_map='auto'),
            batch_padding=False, # if false, inference with for-loop without batch padding
            run_cfg=dict(num_gpus=1, num_procs=1),
        )
        baichuan2_13b_chat_op_models.append(baichuan2_model_13b)
        if i == 1:
            baichuan2_13b_chat_op_model_test = [baichuan2_model_13b]
    
    
# baichuan2-13b-chat_{A/L}
baichuan2_13b_chat_op_general_path = [522,562,614,716,818,920,1022]    
baichuan2_13b_chat_op_general_models = []    
for i in range(1, len(baichuan2_13b_chat_op_general_path)+1):    
    baichuan2_model_13b = dict(
        type=HuggingFaceCausalLM,
        abbr=f'baichuan2-13b-chat-op-general-v{i}',
        path=f"/ArithmeticGPT/sft/models/baichuan2_13b_chat_op_general_cal_config_v{i}/checkpoint-{baichuan2_13b_chat_op_general_path[i-1]}",
        tokenizer_path=f"/ArithmeticGPT/sft/models/baichuan2_13b_chat_op_general_cal_config_v{i}/checkpoint-{baichuan2_13b_chat_op_general_path[i-1]}",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              trust_remote_code=True,
                              ),
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=512,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
    baichuan2_13b_chat_op_general_models.append(baichuan2_model_13b)
    
# baichuan2-13b-chat_{A/L/M}
baichuan2_13b_chat_op_general_mwp_path = [624, 664, 716, 1124, 1634]    
baichuan2_13b_chat_op_general_mwp_models = []    
for i in range(1, len(baichuan2_13b_chat_op_general_mwp_path)+1):    
    baichuan2_model_13b = dict(
        type=HuggingFaceCausalLM,
        abbr=f'baichuan2-13b-chat-op-general-wp-v{i}',
        path=f"/ArithmeticGPT/sft/models/baichuan2_13b_chat_op_general_wp_cal_config_v{i}/checkpoint-{baichuan2_13b_chat_op_general_mwp_path[i-1]}",
        tokenizer_path=f"/ArithmeticGPT/sft/models/baichuan2_13b_chat_op_general_wp_cal_config_v{i}/checkpoint-{baichuan2_13b_chat_op_general_mwp_path[i-1]}",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              trust_remote_code=True,
                              ),
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=512,
        model_kwargs=dict(trust_remote_code=True, device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
    baichuan2_13b_chat_op_general_mwp_models.append(baichuan2_model_13b)
    
baichuan2_13b_chat_tool = baichuan2_13b_chat_op_models + baichuan2_13b_chat_op_general_models + baichuan2_13b_chat_op_general_mwp_models