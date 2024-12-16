from mmengine.config import read_base

with read_base():
   # math datasets
   ## llama 
    from ..arith_std.arith_std_gen_zero_llama2 import arith_std_datasets as arith_std_datasets_llama2_zero
    from ..mawps.mawps_gen_zero_llama2 import mawps_datasets as mawps_datasets_llama2_zero
    from ..asdiv_a.asdiv_a_gen_zero_llama2 import asdiv_datasets as asdiv_datasets_llama2_zero
    from ..gsm8k.gsm8k_gen_zero_llama2 import gsm8k_datasets as gsm8k_datasets_llama2_zero
    from ..svamp.svamp_gen_zero_llama2 import svamp_datasets as svamp_datasets_llama2_zero
    from ..math401.math401_gen_zero_llama2 import math401_datasets as math401_datasets_llama2_zero
    from ..bbh.bb_Arithmetic_gen_zero_llama2 import bb_arith_datasets as bb_arith_datasets_llama2_zero
    
    ## baichuan
    from ..arith_std.arith_std_gen_zero_baichuan2 import arith_std_datasets as arith_std_datasets_baichuan2_zero
    from ..mawps.mawps_gen_zero_baichuan2 import mawps_datasets as mawps_datasets_baichuan2_zero
    from ..asdiv_a.asdiv_a_gen_zero_baichuan2 import asdiv_datasets as asdiv_datasets_baichuan2_zero
    from ..gsm8k.gsm8k_gen_zero_baichuan2 import gsm8k_datasets as gsm8k_datasets_baichuan2_zero
    from ..svamp.svamp_gen_zero_baichuan2 import svamp_datasets as svamp_datasets_baichuan2_zero
    from ..math401.math401_gen_zero_baichuan2 import math401_datasets as math401_datasets_baichuan2_zero
    from ..bbh.bb_Arithmetic_gen_zero_baichuan2 import bb_arith_datasets as bb_arith_datasets_baichuan2_zero
    
    
    

    
 
datasets_math_llama2_zeroshot = [*arith_std_datasets_llama2_zero,*mawps_datasets_llama2_zero,*asdiv_datasets_llama2_zero,
                   *gsm8k_datasets_llama2_zero,*svamp_datasets_llama2_zero,*math401_datasets_llama2_zero,*bb_arith_datasets_llama2_zero]


datasets_math_baichuan2_zeroshot = [*arith_std_datasets_baichuan2_zero,*mawps_datasets_baichuan2_zero,*asdiv_datasets_baichuan2_zero,
                   *gsm8k_datasets_baichuan2_zero,*svamp_datasets_baichuan2_zero,*math401_datasets_baichuan2_zero,*bb_arith_datasets_baichuan2_zero]
   
