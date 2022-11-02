import torch
import numpy as np
import torch.distributed as dist
import random
import os


def int_random_seed(seed=0, device="cuda", rank=0, world_size=1):
    """got random seed for different case"""
    if seed is not None:
        return seed
    
    '''initialize random seed'''
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed
    
    '''multigpu  make sure all ranks share the same random seed'''
    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item

def set_random_seed(seed, deterministic=False):
    """set random seed for all function"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
def collate_fn(batch):
    return

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)