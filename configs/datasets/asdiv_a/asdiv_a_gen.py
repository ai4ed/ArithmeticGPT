from mmengine.config import read_base

with read_base():
    from .asdiv_a_gen_zero import asdiv_datasets  # noqa: F401, F403
    from .asdiv_a_gen_five import asdiv_datasets_five