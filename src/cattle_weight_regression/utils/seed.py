"""Utilities for deterministic training."""

import random

import numpy as np
import torch


def seed_data_rng(seed: int) -> torch.Generator:
    """Seed all data-pipeline RNG sources and return a seeded DataLoader generator.

    Controls batch ordering and data augmentation. Call this before building
    datasets and DataLoaders. Also applies cuDNN determinism flags which must
    be set before any CUDA kernels are launched.

    Returns a torch.Generator to pass as the `generator` argument to DataLoader.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Force cuDNN to use deterministic algorithms; disables the autotuner that
    # picks the fastest (potentially non-deterministic) convolution kernel.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_model_rng(seed: int) -> None:
    """Seed PyTorch for deterministic model weight initialisation.

    Call this immediately before model creation so the global PyTorch RNG
    state is fixed at the point weights are drawn.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:  # noqa: ARG001
    """Re-seed each DataLoader worker so augmentation is deterministic.

    PyTorch derives each worker's base seed from the parent generator, but
    NumPy and Python random are not automatically seeded — this function
    synchronises them so all three RNG sources agree.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
