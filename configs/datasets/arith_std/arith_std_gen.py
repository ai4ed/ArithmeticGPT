from mmengine.config import read_base

with read_base():
    from .arith_std_gen_zero import arith_std_datasets  # noqa: F401, F403
    from .arith_std_gen_five import arith_std_datasets_five