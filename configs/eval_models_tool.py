from mmengine.config import read_base
from opencompass.runners import SlurmRunner, LocalRunner
from opencompass.tasks import OpenICLInferTask


with read_base():
    
    # math datasets
    from .datasets.collections.math_datasets_tool_zeroshot import datasets_math_llama2_zeroshot, datasets_math_baichuan2_zeroshot,arith_std_datasets_llama2_zero
    # general language datasets
    from .datasets.collections.llama2_chat_general_datasets import llama2_chat_general_datasets, BoolQ_datasets_llama2_zero
    from .datasets.collections.baichuan2_13b_chat_general_datasets import baichuan2_chat_general_datasets, BoolQ_datasets_baichuan2_zero

    # models
    from .models. baseline_models import base_models # base models
    from .models.llm_tool import llama2_13b_chat_tool, llama2_70b_chat_tool, baichuan2_13b_chat_tool, llama2_13b_chat_op_model_test, baichuan2_13b_chat_op_model_test #ArithmeticGPT

models = baichuan2_13b_chat_op_model_test
datasets = datasets_math_baichuan2_zeroshot + baichuan2_chat_general_datasets
# datasets = [*arith_std_datasets_llama2_zero, *BoolQ_datasets_llama2_zero]

## slurm
infer = dict(
    runner=dict(
        type=SlurmRunner,
        task=dict(type=OpenICLInferTask),  
        max_num_workers=112,  # Maximum number of simultaneous evaluation tasks
        retry=2,  # The number of retries of failed tasks to avoid unexpected errors
    ),
)