"""Apply interpreter-startup settings before the training entry point runs."""

import os
import sys


def prepare_training_process(seed: int, cublas_workspace_config: str) -> None:
    """Restart Python when needed so hash randomisation uses the configured seed.

    Only the training CLI calls this helper. The replacement process inherits
    the working directory and arguments, but starts with the configured values.
    """
    required = {
        "PYTHONHASHSEED": str(int(seed)),
        "CUBLAS_WORKSPACE_CONFIG": str(cublas_workspace_config),
    }
    if all(os.environ.get(key) == value for key, value in required.items()):
        return
    environment = os.environ.copy()
    environment.update(required)
    os.execve(
        sys.executable,
        [sys.executable, "-m", "src.itd_tcnTransformer.scripts.06_train_model", *sys.argv[1:]],
        environment,
    )
