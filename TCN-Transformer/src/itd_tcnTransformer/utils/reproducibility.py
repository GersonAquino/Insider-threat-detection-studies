import os
import random

import numpy as np
import torch


__all__ = [
    "build_reproducibility_report",
    "create_torch_generator",
    "seed_worker",
    "set_reproducibility",
]


def build_reproducibility_report(
    seed: int,
    deterministic: bool,
    warn_only: bool,
    cudnn_benchmark: bool,
    cublas_workspace_config: str,
) -> dict[str, object]:
    return {
        "random_seed": int(seed),
        "deterministic_training": bool(deterministic),
        "deterministic_warn_only": bool(warn_only),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(cudnn_benchmark),
        "cublas_workspace_config": str(cublas_workspace_config),
        "torch_initial_seed": int(torch.initial_seed()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "torch_version": str(torch.__version__),
    }


def create_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return generator


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_reproducibility(
    seed: int,
    deterministic: bool = True,
    warn_only: bool = True,
    cudnn_benchmark: bool = False,
    cublas_workspace_config: str = ":4096:8",
) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(cublas_workspace_config)

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
