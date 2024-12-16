from mmengine.config import read_base

with read_base():
    
    from ..SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl_314797 import BoolQ_datasets as BoolQ_datasets_baichuan2_zero
    # from ..race.race_gen_69ee4f import race_datasets as race_datasets
    from ..race.race_ppl_5831a0 import race_datasets as race_datasets_baichuan2_zero
    from ..hellaswag.hellaswag_ppl_47bff9 import hellaswag_datasets as hellaswag_datasets_baichuan2_zero
    from ..piqa.piqa_ppl_1cf9f0 import piqa_datasets as piqa_datasets_baichuan2_zero 
    
    
baichuan2_chat_general_datasets = [*BoolQ_datasets_baichuan2_zero, *race_datasets_baichuan2_zero, *hellaswag_datasets_baichuan2_zero, *piqa_datasets_baichuan2_zero]