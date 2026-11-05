import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)


def get_rng(seed: int = SEED) -> np.random.Generator:
    """Preferred: an explicit NumPy random generator."""
    return np.random.default_rng(seed)
