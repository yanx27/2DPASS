import numpy as np

__all__ = ['cosine_schedule_with_warmup']


def cosine_schedule_with_warmup(k, num_epochs, batch_size, dataset_size, num_gpu):
    batch_size *= num_gpu

    if num_gpu == 1:
        warmup_iters = 0
    else:
        warmup_iters = 1000 // num_gpu

    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
        return 0.5 * (1 + np.cos(np.pi * (k - warmup_iters) / (num_epochs * iter_per_epoch)))



