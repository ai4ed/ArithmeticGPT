from mmengine.config import read_base

with read_base():
    from .mawps_gen_zero import mawps_datasets  # noqa: F401, F403
    from .mawps_gen_five import mawps_datasets_five