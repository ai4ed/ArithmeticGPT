from mmengine.config import read_base

with read_base():

    from ..SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl_314797 import BoolQ_datasets as BoolQ_datasets_llama2_zero
    from ..race.race_ppl_5831a0 import race_datasets as race_datasets_llama2_zero
    from ..hellaswag.hellaswag_ppl_a6e128 import hellaswag_datasets as hellaswag_datasets_llama2_zero
    from ..piqa.piqa_ppl_0cfff2 import piqa_datasets as piqa_datasets_llama2_zero
    

llama2_chat_general_datasets = [*BoolQ_datasets_llama2_zero, *race_datasets_llama2_zero, *hellaswag_datasets_llama2_zero, *piqa_datasets_llama2_zero]
