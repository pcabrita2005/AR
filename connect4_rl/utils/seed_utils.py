import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    # Python's random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (CUDA if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Environment variable for hash randomization (Python 3.3+)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional: Enable deterministic algorithms (may reduce performance)
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # For CUDA 10.2+

    print(f"✓ All random seeds set to {seed}")
