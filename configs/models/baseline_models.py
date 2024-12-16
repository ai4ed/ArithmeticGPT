from opencompass.models import HuggingFaceCausalLM
from opencompass.models import HuggingFace

all_models = ['Llama-2-7b-chat-hf','Llama-2-13b-chat-hf','Llama-2-70b-chat-hf','Baichuan2-13B-Chat']
base_models = []
for model_name in all_models:
    if '70' in model_name:
        num_gpu = 2
    else:
        num_gpu = 1
    temp_config = dict(
                        type=HuggingFaceCausalLM,
                        abbr=f'{model_name}',
                        path=f"/ArithmeticGPT/sft/models/{model_name}",
                        tokenizer_path=f'/ArithmeticGPT/sft/models/{model_name}',
                        tokenizer_kwargs=dict(padding_side='left',
                                            truncation_side='left',
                                            use_fast=False,
                                            trust_remote_code=True,
                                            ),
                        max_out_len=512,
                        max_seq_len=2048,
                        batch_size=8,
                        model_kwargs=dict(trust_remote_code=True, 
                                          device_map='auto',
                                          ),
                        batch_padding=True, # if false, inference with for-loop without batch padding
                        run_cfg=dict(num_gpus=num_gpu, num_procs=1),
                    )
    base_models.append(temp_config)