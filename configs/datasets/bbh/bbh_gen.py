from mmengine.config import read_base

with read_base():
    from .bbh_gen_five import bbh_datasets  # noqa: F401, F403
    # from .bb_Arithmetic_gen_v1 import bb_arith_datasets  # noqa: F401, F403
