from mmengine.config import read_base

with read_base():
    from .math401_gen_few import math401_datasets as math401_datasets_few  # noqa: F401, F403
    from .math401_gen_zero import math401_datasets  # noqa: F401, F403
