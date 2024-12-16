from mmengine.config import read_base

with read_base():
    from .gsm8k_gen_few import gsm8k_datasets as gsm8k_datasets_few  # noqa: F401, F403
    from .gsm8k_gen_zero import gsm8k_datasets  # noqa: F401, F403

