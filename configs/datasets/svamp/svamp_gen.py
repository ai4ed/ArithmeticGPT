from mmengine.config import read_base

with read_base():
    from .svamp_gen_few import svamp_datasets as svamp_datasets_few  # noqa: F401, F403
    from .svamp_gen_zero import svamp_datasets  # noqa: F401, F403
